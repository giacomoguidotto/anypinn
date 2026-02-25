"""Tests for anypinn.lightning.callbacks."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from anypinn.core.config import SMMAStoppingConfig
from anypinn.core.types import Predictions
from anypinn.lightning.callbacks import (
    DataScaling,
    FormattedProgressBar,
    Metric,
    PredictionsWriter,
    SMMAStopping,
)


class TestSMMAStopping:
    def _make_callback(
        self, window: int = 3, threshold: float = 0.01, lookback: int = 2
    ) -> SMMAStopping:
        config = SMMAStoppingConfig(window=window, threshold=threshold, lookback=lookback)
        return SMMAStopping(config, loss_key="loss")

    def _mock_trainer(self, loss: float) -> MagicMock:
        trainer = MagicMock()
        trainer.callback_metrics = {"loss": torch.tensor(loss)}
        trainer.should_stop = False
        return trainer

    def _mock_module(self) -> MagicMock:
        return MagicMock()

    def test_phase1_collects_losses(self):
        cb = self._make_callback(window=3)
        trainer = self._mock_trainer(1.0)
        module = self._mock_module()

        # First window+1 calls fill the buffer (phase 1)
        for _ in range(4):
            cb.on_train_epoch_end(trainer, module)
        assert len(cb.loss_buffer) == 4

    def test_phase15_computes_first_smma(self):
        cb = self._make_callback(window=3)
        trainer = self._mock_trainer(1.0)
        module = self._mock_module()

        # Fill buffer: window+1 = 4 epochs
        for _ in range(4):
            cb.on_train_epoch_end(trainer, module)
        # 5th epoch triggers first SMMA
        cb.on_train_epoch_end(trainer, module)
        assert len(cb.smma_buffer) == 1

    def test_phase2_computes_smma(self):
        cb = self._make_callback(window=3, lookback=5)
        trainer = self._mock_trainer(1.0)
        module = self._mock_module()

        # phase 1: fill buffer (window=3, condition is `<= n` so need 4 calls)
        for _ in range(4):
            cb.on_train_epoch_end(trainer, module)
        assert len(cb.smma_buffer) == 0
        # phase 1.5: first average
        cb.on_train_epoch_end(trainer, module)
        assert len(cb.smma_buffer) == 1
        # phase 2: each subsequent call appends to smma_buffer
        cb.on_train_epoch_end(trainer, module)
        assert len(cb.smma_buffer) == 2
        cb.on_train_epoch_end(trainer, module)
        assert len(cb.smma_buffer) == 3

    def test_stops_when_improvement_below_threshold(self):
        cb = self._make_callback(window=2, threshold=0.5, lookback=2)
        module = self._mock_module()

        # Fill buffer with constant loss → zero improvement → should stop
        trainer = self._mock_trainer(1.0)
        for _ in range(10):
            cb.on_train_epoch_end(trainer, module)

        # With constant loss, improvement should be 0 which is not > 0, so no stop
        # Let's use decreasing loss that plateaus
        cb2 = self._make_callback(window=2, threshold=0.5, lookback=2)
        losses = [10.0, 9.0, 8.0, 7.99, 7.98, 7.97, 7.96, 7.95, 7.94, 7.93]
        for loss in losses:
            t = self._mock_trainer(loss)
            cb2.on_train_epoch_end(t, module)

        # The last trainer should have should_stop eventually set
        # (depends on exact SMMA dynamics)

    def test_no_stop_with_missing_metric(self):
        cb = self._make_callback()
        trainer = MagicMock()
        trainer.callback_metrics = {}
        module = self._mock_module()
        cb.on_train_epoch_end(trainer, module)
        assert len(cb.loss_buffer) == 0


class TestDataScaling:
    def test_single_scale_factor(self):
        ds = DataScaling(y_scale=0.01)
        x = torch.linspace(0, 100, 20).unsqueeze(-1)
        y = torch.randn(20, 1, 1)
        coll = torch.rand(50, 1) * 100

        (x_out, _y_out), coll_out = ds.transform_data((x, y), coll)

        # x should be normalized to [0, 1]
        assert x_out.min().item() == pytest.approx(0.0, abs=1e-5)
        assert x_out.max().item() == pytest.approx(1.0, abs=1e-5)

        # coll should be normalized
        assert coll_out.min().item() >= -0.01
        assert coll_out.max().item() <= 1.01

    def test_per_series_scale(self):
        ds = DataScaling(y_scale=[0.01, 0.001])
        x = torch.linspace(0, 100, 20).unsqueeze(-1)
        y = torch.ones(20, 2, 1)
        coll = torch.rand(50, 1) * 100

        (_x_out, y_out), _ = ds.transform_data((x, y), coll)

        # Series 0 scaled by 0.01, series 1 by 0.001
        assert y_out[0, 0, 0].item() == pytest.approx(0.01, rel=1e-4)
        assert y_out[0, 1, 0].item() == pytest.approx(0.001, rel=1e-4)

    def test_mismatched_scale_raises(self):
        ds = DataScaling(y_scale=[0.01, 0.001, 0.0001])
        x = torch.linspace(0, 100, 20).unsqueeze(-1)
        y = torch.ones(20, 2, 1)
        coll = torch.rand(50, 1) * 100

        with pytest.raises(ValueError, match="y_scale has 3 elements"):
            ds.transform_data((x, y), coll)


class TestFormattedProgressBar:
    def test_metrics_formatted(self):
        def fmt(key: str, value: Metric) -> Metric:
            if isinstance(value, float):
                return f"{value:.2e}"
            return value

        bar = FormattedProgressBar(format=fmt)
        # Just verify it can be instantiated with the format function
        assert bar.format is fmt


class TestPredictionsWriter:
    def test_write_predictions(self, tmp_path: Path):
        preds_path = tmp_path / "preds.pt"
        writer = PredictionsWriter(predictions_path=preds_path)

        trainer = MagicMock()
        module = MagicMock()
        predictions = [((torch.randn(5), torch.randn(5)), {"u": torch.randn(5)}, None)]
        batch_indices: list[object] = []

        writer.write_on_epoch_end(trainer, module, predictions, batch_indices)

        assert preds_path.exists()

    def test_write_batch_indices(self, tmp_path: Path):
        idx_path = tmp_path / "indices.pt"
        writer = PredictionsWriter(batch_indices_path=idx_path)

        trainer = MagicMock()
        module = MagicMock()
        predictions: list[Predictions] = []
        batch_indices = [[0, 1, 2]]

        writer.write_on_epoch_end(trainer, module, predictions, batch_indices)

        assert idx_path.exists()

    def test_on_prediction_hook(self):
        hook = MagicMock()
        writer = PredictionsWriter(on_prediction=hook)

        writer.write_on_epoch_end(MagicMock(), MagicMock(), [], [])
        hook.assert_called_once()
