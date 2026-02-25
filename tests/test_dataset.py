"""Tests for anypinn.core.dataset — PINNDataset and PINNDataModule."""

import pytest
import torch
from typing_extensions import override

from anypinn.core.config import GenerationConfig, MLPConfig, PINNHyperparameters, ScalarConfig
from anypinn.core.dataset import PINNDataModule, PINNDataset
from anypinn.core.types import DataBatch

_HP = PINNHyperparameters(
    lr=1e-3,
    training_data=GenerationConfig(
        batch_size=16,
        data_ratio=0.5,
        collocations=50,
        x=torch.linspace(0, 1, 40).unsqueeze(-1),
        noise_level=0.0,
        args_to_train={},
    ),
    fields_config=MLPConfig(in_dim=2, out_dim=1, hidden_layers=[8], activation="tanh"),
    params_config=ScalarConfig(init_value=1.0),
)


class _Dummy2DDataModule(PINNDataModule):
    """Concrete DataModule with 2-D spatial inputs for testing."""

    @override
    def gen_data(self, config: GenerationConfig) -> DataBatch:
        x = torch.rand(40, 2)
        y = torch.rand(40, 1)
        return x, y


class TestPINNDataset:
    def _make_dataset(
        self,
        n_data: int = 20,
        n_coll: int = 100,
        batch_size: int = 16,
        data_ratio: float | int = 0.5,
    ) -> PINNDataset:
        x_data = torch.linspace(0, 10, n_data).unsqueeze(-1)
        y_data = torch.randn(n_data, 1, 1)
        x_coll = torch.rand(n_coll, 1) * 10
        return PINNDataset(x_data, y_data, x_coll, batch_size, data_ratio)

    def test_len_float_ratio(self):
        ds = self._make_dataset(n_data=20, batch_size=16, data_ratio=0.5)
        # K = round(0.5 * 16) = 8, len = ceil(20 / 8) = 3
        assert ds.K == 8
        assert len(ds) == 3

    def test_len_int_ratio(self):
        ds = self._make_dataset(n_data=20, batch_size=16, data_ratio=4)
        # K = 4, len = ceil(20 / 4) = 5
        assert ds.K == 4
        assert len(ds) == 5

    def test_getitem_shapes(self):
        ds = self._make_dataset(n_data=20, batch_size=16, data_ratio=0.5)
        (x_data, y_data), x_coll = ds[0]
        assert x_data.shape == (8, 1)
        assert y_data.shape == (8, 1, 1)
        assert x_coll.shape == (8, 1)  # C = batch_size - K = 16 - 8 = 8

    def test_data_wraparound(self):
        ds = self._make_dataset(n_data=10, batch_size=16, data_ratio=0.5)
        # K = 8, last batch starts at index 8, needs 8 points but only 2 remain → wraps
        (x_data, _), _ = ds[1]
        assert x_data.shape[0] == 8

    def test_collocation_deterministic(self):
        ds = self._make_dataset()
        (_, _), coll_a = ds[0]
        (_, _), coll_b = ds[0]
        assert torch.equal(coll_a, coll_b)

    def test_collocation_different_indices(self):
        ds = self._make_dataset()
        (_, _), coll_0 = ds[0]
        (_, _), coll_1 = ds[1]
        assert not torch.equal(coll_0, coll_1)

    def test_data_ratio_zero(self):
        ds = self._make_dataset(batch_size=16, data_ratio=0.0)
        assert ds.K == 0
        assert ds.C == 16

    def test_data_ratio_one(self):
        ds = self._make_dataset(batch_size=16, data_ratio=1.0)
        assert ds.K == 16
        assert ds.C == 0

    def test_batch_size_zero_raises(self):
        with pytest.raises(ValueError, match="batch_size must be positive"):
            PINNDataset(
                torch.randn(10, 1),
                torch.randn(10, 1),
                torch.randn(50, 1),
                batch_size=0,
                data_ratio=0.5,
            )

    def test_invalid_float_ratio_raises(self):
        with pytest.raises(ValueError, match="Float data_ratio must be in"):
            PINNDataset(
                torch.randn(10, 1),
                torch.randn(10, 1),
                torch.randn(50, 1),
                batch_size=16,
                data_ratio=1.5,
            )

    def test_pinndata_accepts_multidim_x(self):
        """PINNDataset must not reject x/coll with d > 1."""
        x = torch.rand(100, 2)
        y = torch.rand(100, 3, 1)
        coll = torch.rand(200, 2)
        ds = PINNDataset(x_data=x, y_data=y, x_coll=coll, batch_size=10, data_ratio=0.5)
        (x_b, _y_b), coll_b = ds[0]
        assert x_b.shape[1] == 2
        assert coll_b.shape[1] == 2


class TestPINNDataModuleSetup:
    def _make_dm(self) -> _Dummy2DDataModule:
        return _Dummy2DDataModule(hp=_HP)

    def test_setup_succeeds_with_2d_data(self):
        """setup() must not raise when x_data and coll both have d=2."""
        dm = self._make_dm()
        dm.setup()

    def test_context_domain_has_correct_ndim(self):
        """InferredContext.domain.ndim must equal the input spatial dimension."""
        dm = self._make_dm()
        dm.setup()
        assert dm.context.domain.ndim == 2

    def test_pinn_dataset_batch_shapes_are_multidim(self):
        """Batches produced by the dataset must preserve d=2 in x and coll."""
        dm = self._make_dm()
        dm.setup()
        (x_b, _), coll_b = dm.pinn_ds[0]
        assert x_b.shape[1] == 2
        assert coll_b.shape[1] == 2

    def test_sampler_matches_domain_dimensions(self):
        """Collocation sampler output must match the data dimensionality."""
        dm = self._make_dm()
        dm.setup()
        assert dm.coll.shape[1] == 2
        assert dm.coll.shape[0] == 50

    def test_context_before_setup_raises(self):
        """Accessing context before setup() must raise RuntimeError, not AttributeError."""
        dm = self._make_dm()
        with pytest.raises(RuntimeError, match="setup\\(\\)"):
            _ = dm.context
