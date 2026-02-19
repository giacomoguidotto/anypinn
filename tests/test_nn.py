"""Tests for anypinn.core.nn — Field, Parameter, Argument, Domain1D."""

import pytest
import torch
from torch import Tensor
import torch.nn as nn

from anypinn.core.config import MLPConfig
from anypinn.core.nn import Argument, Domain1D, Field, Parameter, get_activation
from anypinn.core.types import Activations

# ── Domain1D ──────────────────────────────────────────────────────────


class TestDomain1D:
    def test_from_x_basic(self):
        x = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0]).unsqueeze(-1)
        d = Domain1D.from_x(x)
        assert d.x0 == pytest.approx(0.0)
        assert d.x1 == pytest.approx(2.0)
        assert d.dx == pytest.approx(0.5)

    def test_from_x_two_points(self):
        x = torch.tensor([3.0, 7.0]).unsqueeze(-1)
        d = Domain1D.from_x(x)
        assert d.x0 == pytest.approx(3.0)
        assert d.x1 == pytest.approx(7.0)
        assert d.dx == pytest.approx(4.0)

    def test_from_x_single_point_raises(self):
        x = torch.tensor([5.0]).unsqueeze(-1)
        with pytest.raises(ValueError, match="At least two points"):
            Domain1D.from_x(x)

    def test_repr(self):
        d = Domain1D(x0=0.0, x1=1.0, dx=0.1)
        assert "Domain1D" in repr(d)
        assert "0.0" in repr(d)
        assert "1.0" in repr(d)


# ── get_activation ────────────────────────────────────────────────────


ALL_ACTIVATIONS: list[Activations] = [
    "tanh",
    "relu",
    "leaky_relu",
    "sigmoid",
    "selu",
    "softplus",
    "identity",
]


@pytest.mark.parametrize("name", ALL_ACTIVATIONS)
def test_get_activation_returns_module(name: Activations):
    act = get_activation(name)
    assert isinstance(act, nn.Module)
    # Smoke test: should accept a tensor
    out = act(torch.randn(4))
    assert out.shape == (4,)


# ── Field ─────────────────────────────────────────────────────────────


class TestField:
    def test_forward_shape(self, field: Field, x_tensor: Tensor):
        out = field(x_tensor)
        assert out.shape == (50, 1)

    def test_forward_shape_multi_output(self, multi_field: Field, x_tensor: Tensor):
        out = multi_field(x_tensor)
        assert out.shape == (50, 3)

    def test_xavier_init(self, field: Field):
        for m in field.modules():
            if isinstance(m, nn.Linear):
                assert m.bias is not None
                assert torch.all(m.bias == 0.0)

    def test_output_activation(self):
        config = MLPConfig(
            in_dim=1,
            out_dim=1,
            hidden_layers=[8],
            activation="tanh",
            output_activation="sigmoid",
        )
        f = Field(config)
        out = f(torch.randn(10, 1))
        # sigmoid output is in (0, 1)
        assert torch.all(out >= 0.0)
        assert torch.all(out <= 1.0)

    def test_encode_function(self):
        config = MLPConfig(
            in_dim=2,
            out_dim=1,
            hidden_layers=[8],
            activation="tanh",
            encode=lambda x: torch.cat([x, x**2], dim=-1),
        )
        f = Field(config)
        out = f(torch.randn(5, 1))
        assert out.shape == (5, 1)

    def test_gradients_flow(self, field: Field):
        x = torch.randn(5, 1, requires_grad=True)
        out = field(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# ── Argument ──────────────────────────────────────────────────────────


class TestArgument:
    def test_float_value(self):
        arg = Argument(3.14)
        x = torch.randn(5, 1)
        result = arg(x)
        assert result.item() == pytest.approx(3.14)

    def test_callable_value(self):
        arg = Argument(lambda x: x * 2)
        x = torch.tensor([1.0, 2.0, 3.0])
        result = arg(x)
        assert torch.allclose(result, torch.tensor([2.0, 4.0, 6.0]))

    def test_float_caches_per_device(self):
        arg = Argument(1.0)
        x = torch.randn(3)
        _ = arg(x)
        assert len(arg._tensor_cache) == 1

    def test_repr(self):
        arg = Argument(2.5)
        assert "2.5" in repr(arg)


# ── Parameter ─────────────────────────────────────────────────────────


class TestParameter:
    def test_scalar_mode(self, scalar_param: Parameter):
        assert scalar_param.mode == "scalar"
        out = scalar_param()
        assert out.item() == pytest.approx(0.5)

    def test_scalar_expand(self, scalar_param: Parameter):
        x = torch.randn(5, 1)
        out = scalar_param(x)
        assert out.shape == x.shape

    def test_mlp_mode(self, mlp_param: Parameter):
        assert mlp_param.mode == "mlp"
        x = torch.randn(5, 1)
        out = mlp_param(x)
        assert out.shape == (5, 1)

    def test_mlp_requires_input(self, mlp_param: Parameter):
        with pytest.raises(TypeError):
            mlp_param(None)

    def test_scalar_is_learnable(self, scalar_param: Parameter):
        params = list(scalar_param.parameters())
        assert len(params) == 1
        assert params[0].requires_grad

    def test_mlp_xavier_init(self, mlp_param: Parameter):
        for m in mlp_param.modules():
            if isinstance(m, nn.Linear):
                assert m.bias is not None
                assert torch.all(m.bias == 0.0)
