"""Tests for anypinn.core.validation — resolve_validation, ColumnRef."""

from pathlib import Path

import pytest
import torch

from anypinn.core.validation import ColumnRef, resolve_validation


class TestResolveValidation:
    def test_empty_registry(self):
        result = resolve_validation({})
        assert result == {}

    def test_none_source_skipped(self):
        result = resolve_validation({"param": None})
        assert result == {}

    def test_callable_passthrough(self):
        def fn(x):
            return x * 2

        result = resolve_validation({"param": fn})
        assert "param" in result
        x = torch.tensor([1.0, 2.0])
        assert torch.allclose(result["param"](x), torch.tensor([2.0, 4.0]))

    def test_column_ref_without_df_path_raises(self):
        registry = {"beta": ColumnRef(column="Rt")}
        with pytest.raises(ValueError, match="no df_path"):
            resolve_validation(registry, df_path=None)

    def test_column_ref_missing_column_raises(self, tmp_path: Path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("a,b\n1,2\n3,4\n")
        registry = {"beta": ColumnRef(column="missing_col")}
        with pytest.raises(ValueError, match="not found in data"):
            resolve_validation(registry, df_path=csv_path)

    def test_column_ref_resolved(self, tmp_path: Path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("t,beta_true\n0,0.3\n1,0.4\n2,0.5\n")
        registry = {"beta": ColumnRef(column="beta_true")}
        result = resolve_validation(registry, df_path=csv_path)

        assert "beta" in result
        # Lookup by index: idx 0 -> 0.3, idx 1 -> 0.4
        x = torch.tensor([[0.0], [1.0]])
        vals = result["beta"](x)
        assert vals[0].item() == pytest.approx(0.3, abs=1e-4)
        assert vals[1].item() == pytest.approx(0.4, abs=1e-4)

    def test_column_ref_with_transform(self, tmp_path: Path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("t,val\n0,10.0\n1,20.0\n")
        registry = {"p": ColumnRef(column="val", transform=lambda v: v / 10.0)}
        result = resolve_validation(registry, df_path=csv_path)

        x = torch.tensor([[0.0]])
        val = result["p"](x)
        assert val.item() == pytest.approx(1.0, abs=1e-4)

    def test_mixed_registry(self, tmp_path: Path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("t,col\n0,1.0\n1,2.0\n")
        registry = {
            "a": lambda x: x,
            "b": ColumnRef(column="col"),
            "c": None,
        }
        result = resolve_validation(registry, df_path=csv_path)
        assert "a" in result
        assert "b" in result
        assert "c" not in result
