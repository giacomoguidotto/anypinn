"""Tests for anypinn.lib.diff — differential operator utilities."""

import pytest
import torch
from torch import Tensor

from anypinn.lib.diff import divergence, grad, hessian, laplacian, mixed_partial, partial

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_x(n: int, d: int) -> Tensor:
    """Random coordinates with requires_grad=True."""
    return torch.randn(n, d, requires_grad=True)


# ---------------------------------------------------------------------------
# grad
# ---------------------------------------------------------------------------


class TestGrad:
    def test_shape(self):
        x = _make_x(16, 3)
        u = (x[:, 0] ** 2).unsqueeze(-1)  # u = x0^2
        g = grad(u, x)
        assert g.shape == (16, 3)

    def test_linear_function(self):
        """u = 3*x0 + 2*x1 => ∇u = [3, 2]."""
        x = _make_x(32, 2)
        u = 3.0 * x[:, 0] + 2.0 * x[:, 1]
        g = grad(u, x)
        expected = torch.tensor([[3.0, 2.0]]).expand(32, 2)
        assert torch.allclose(g, expected, atol=1e-6)

    def test_1d_input_shape(self):
        """u shape (n,) should work the same as (n, 1)."""
        x = _make_x(8, 2)
        u_flat = x[:, 0] ** 2 + x[:, 1] ** 2
        u_col = u_flat.unsqueeze(-1)
        g_flat = grad(u_flat, x)
        g_col = grad(u_col, x)
        assert torch.allclose(g_flat, g_col, atol=1e-6)

    def test_create_graph_false_detaches(self):
        x = _make_x(4, 2)
        u = x.sum()
        g = grad(u, x, create_graph=False)
        assert not g.requires_grad


# ---------------------------------------------------------------------------
# partial
# ---------------------------------------------------------------------------


class TestPartial:
    def test_first_order(self):
        """u = x0^2 * x1 => ∂u/∂x0 = 2*x0*x1."""
        x = _make_x(20, 2)
        u = x[:, 0] ** 2 * x[:, 1]
        du_dx0 = partial(u, x, dim=0)
        expected = (2.0 * x[:, 0] * x[:, 1]).unsqueeze(-1)
        assert du_dx0.shape == (20, 1)
        assert torch.allclose(du_dx0, expected, atol=1e-5)

    def test_second_order(self):
        """u = x0^3 => ∂²u/∂x0² = 6*x0."""
        x = _make_x(20, 1)
        u = x[:, 0] ** 3
        d2u = partial(u, x, dim=0, order=2)
        expected = (6.0 * x[:, 0]).unsqueeze(-1)
        assert torch.allclose(d2u, expected, atol=1e-4)

    def test_third_order(self):
        """u = x0^4 => ∂³u/∂x0³ = 24*x0."""
        x = _make_x(12, 1)
        u = x[:, 0] ** 4
        d3u = partial(u, x, dim=0, order=3)
        expected = (24.0 * x[:, 0]).unsqueeze(-1)
        assert torch.allclose(d3u, expected, atol=1e-3)

    def test_invalid_order(self):
        x = _make_x(4, 1)
        u = x[:, 0]
        with pytest.raises(ValueError, match="order must be >= 1"):
            partial(u, x, dim=0, order=0)

    def test_create_graph_false(self):
        x = _make_x(4, 2)
        u = x[:, 0] ** 2
        result = partial(u, x, dim=0, create_graph=False)
        assert not result.requires_grad


# ---------------------------------------------------------------------------
# mixed_partial
# ---------------------------------------------------------------------------


class TestMixedPartial:
    def test_single_dim_matches_partial(self):
        """mixed_partial(dims=(0,)) should equal partial(dim=0)."""
        x = _make_x(16, 2)
        u = x[:, 0] ** 2 * x[:, 1]
        mp = mixed_partial(u, x, dims=(0,))
        p = partial(u, x, dim=0)
        assert torch.allclose(mp, p, atol=1e-6)

    def test_cross_derivative(self):
        """u = x0^2 * x1^3 => ∂²u/∂x0∂x1 = 6*x0*x1^2."""
        x = _make_x(20, 2)
        u = x[:, 0] ** 2 * x[:, 1] ** 3
        mp = mixed_partial(u, x, dims=(0, 1))
        expected = (6.0 * x[:, 0] * x[:, 1] ** 2).unsqueeze(-1)
        assert torch.allclose(mp, expected, atol=1e-4)

    def test_symmetry(self):
        """∂²u/∂x0∂x1 should equal ∂²u/∂x1∂x0 for smooth u (Schwarz's theorem)."""
        x = _make_x(20, 2)
        u = torch.sin(x[:, 0]) * torch.cos(x[:, 1])
        mp_01 = mixed_partial(u, x, dims=(0, 1))
        mp_10 = mixed_partial(u, x, dims=(1, 0))
        assert torch.allclose(mp_01, mp_10, atol=1e-5)

    def test_empty_dims_raises(self):
        x = _make_x(4, 2)
        u = x.sum(dim=1)
        with pytest.raises(ValueError, match="at least one dimension"):
            mixed_partial(u, x, dims=())

    def test_create_graph_false(self):
        x = _make_x(4, 2)
        u = x[:, 0] * x[:, 1]
        result = mixed_partial(u, x, dims=(0, 1), create_graph=False)
        assert not result.requires_grad


# ---------------------------------------------------------------------------
# laplacian
# ---------------------------------------------------------------------------


class TestLaplacian:
    def test_quadratic_2d(self):
        """u = x0^2 + x1^2 => ∇²u = 2 + 2 = 4."""
        x = _make_x(16, 2)
        u = x[:, 0] ** 2 + x[:, 1] ** 2
        lap = laplacian(u, x)
        assert lap.shape == (16, 1)
        expected = torch.full((16, 1), 4.0)
        assert torch.allclose(lap, expected, atol=1e-5)

    def test_1d_second_derivative(self):
        """u = x^3 => ∇²u = 6x (1D Laplacian = second derivative)."""
        x = _make_x(12, 1)
        u = x[:, 0] ** 3
        lap = laplacian(u, x)
        expected = (6.0 * x[:, 0]).unsqueeze(-1)
        assert torch.allclose(lap, expected, atol=1e-4)

    def test_3d(self):
        """u = x0^2 + 2*x1^2 + 3*x2^2 => ∇²u = 2 + 4 + 6 = 12."""
        x = _make_x(10, 3)
        u = x[:, 0] ** 2 + 2.0 * x[:, 1] ** 2 + 3.0 * x[:, 2] ** 2
        lap = laplacian(u, x)
        expected = torch.full((10, 1), 12.0)
        assert torch.allclose(lap, expected, atol=1e-4)

    def test_gradient_flows(self):
        """Laplacian result must be differentiable w.r.t. x."""
        x = _make_x(8, 2)
        u = x[:, 0] ** 2 + x[:, 1] ** 2
        lap = laplacian(u, x)
        loss = lap.sum()
        loss.backward()
        assert x.grad is not None

    def test_create_graph_false(self):
        x = _make_x(4, 2)
        u = x[:, 0] ** 2 + x[:, 1] ** 2
        lap = laplacian(u, x, create_graph=False)
        assert not lap.requires_grad


# ---------------------------------------------------------------------------
# divergence
# ---------------------------------------------------------------------------


class TestDivergence:
    def test_constant_field(self):
        """v = [1, 1] => ∇·v = 0."""
        x = _make_x(10, 2)
        v = torch.ones(10, 2)
        # v doesn't depend on x, so divergence should be 0...
        # but v must be connected to x's graph. Use x-dependent v instead.
        v = x * 0.0 + 1.0  # connected to x but constant-valued
        div = divergence(v, x)
        assert div.shape == (10, 1)
        expected = torch.zeros(10, 1)
        assert torch.allclose(div, expected, atol=1e-6)

    def test_identity_field(self):
        """v = [x0, x1] => ∇·v = 1 + 1 = 2."""
        x = _make_x(16, 2)
        v = x  # v_i = x_i
        div = divergence(v, x)
        expected = torch.full((16, 1), 2.0)
        assert torch.allclose(div, expected, atol=1e-5)

    def test_3d(self):
        """v = [x0^2, x1^2, x2^2] => ∇·v = 2*x0 + 2*x1 + 2*x2."""
        x = _make_x(10, 3)
        v = x**2
        div = divergence(v, x)
        expected = 2.0 * x.sum(dim=1, keepdim=True)
        assert torch.allclose(div, expected, atol=1e-5)

    def test_shape_mismatch_raises(self):
        x = _make_x(4, 2)
        v = torch.randn(4, 3)
        with pytest.raises(ValueError, match="they must match"):
            divergence(v, x)

    def test_create_graph_false(self):
        x = _make_x(4, 2)
        v = x**2
        div = divergence(v, x, create_graph=False)
        assert not div.requires_grad


# ---------------------------------------------------------------------------
# hessian
# ---------------------------------------------------------------------------


class TestHessian:
    def test_shape(self):
        x = _make_x(8, 3)
        u = x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2
        H = hessian(u, x)
        assert H.shape == (8, 3, 3)

    def test_diagonal_quadratic(self):
        """u = a*x0^2 + b*x1^2 => H = diag(2a, 2b), off-diagonal = 0."""
        a, b = 3.0, 5.0
        x = _make_x(10, 2)
        u = a * x[:, 0] ** 2 + b * x[:, 1] ** 2
        H = hessian(u, x)
        H00 = H[:, 0, 0]
        H11 = H[:, 1, 1]
        H01 = H[:, 0, 1]
        H10 = H[:, 1, 0]
        assert torch.allclose(H00, torch.full_like(H00, 2.0 * a), atol=1e-4)
        assert torch.allclose(H11, torch.full_like(H11, 2.0 * b), atol=1e-4)
        assert torch.allclose(H01, torch.zeros_like(H01), atol=1e-5)
        assert torch.allclose(H10, torch.zeros_like(H10), atol=1e-5)

    def test_symmetry(self):
        """H should be symmetric for smooth u."""
        x = _make_x(8, 3)
        u = torch.sin(x[:, 0] * x[:, 1]) + x[:, 2] ** 3
        H = hessian(u, x)
        assert torch.allclose(H, H.transpose(1, 2), atol=1e-4)

    def test_trace_equals_laplacian(self):
        """tr(H) = ∇²u."""
        x = _make_x(10, 3)
        u = x[:, 0] ** 2 + 2.0 * x[:, 1] ** 2 + 3.0 * x[:, 2] ** 2
        H = hessian(u, x)
        trace = H.diagonal(dim1=1, dim2=2).sum(dim=1, keepdim=True)
        lap = laplacian(u, x)
        assert torch.allclose(trace, lap, atol=1e-4)

    def test_create_graph_false(self):
        x = _make_x(4, 2)
        u = x[:, 0] ** 2 + x[:, 1] ** 2
        H = hessian(u, x, create_graph=False)
        assert not H.requires_grad
