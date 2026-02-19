"""Tests for anypinn.core.dataset — PINNDataset."""

import pytest
import torch

from anypinn.core.dataset import PINNDataset


class TestPINNDataset:
    def _make_dataset(
        self,
        n_data: int = 20,
        n_coll: int = 100,
        batch_size: int = 16,
        data_ratio: float | int = 0.5,
    ) -> PINNDataset:
        x_data = torch.linspace(0, 10, n_data).unsqueeze(-1)
        y_data = torch.randn(n_data, 1)
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
        assert y_data.shape == (8, 1)
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
