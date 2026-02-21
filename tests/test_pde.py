"""Tests for anypinn.problems.pde — DirichletBCConstraint, NeumannBCConstraint."""

import pytest
import torch
from torch import Tensor
import torch.nn as nn

from anypinn.core.config import MLPConfig
from anypinn.core.nn import Field
from anypinn.problems.pde import BoundaryCondition, DirichletBCConstraint, NeumannBCConstraint

# Minimal dummy batch — BC constraints only use x_coll device, not the data
DUMMY_BATCH = ((torch.zeros(4, 1), torch.zeros(4, 1)), torch.zeros(4, 1))


class _ZeroField(Field):
    """Field that always outputs zeros; used to test zero Dirichlet loss."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.zeros(x.shape[0], 1)


class _LinearField(Field):
    """Field that returns the first coordinate; u(x,y)=x so du/dx=1, du/dy=0."""

    def forward(self, x: Tensor) -> Tensor:
        return x[:, 0:1]


class TestDirichletBCConstraint:
    def _make_field(self) -> Field:
        return Field(MLPConfig(in_dim=2, out_dim=1, hidden_layers=[8], activation="tanh"))

    def _make_bc(self, target: float = 0.0, n_pts: int = 20) -> BoundaryCondition:
        return BoundaryCondition(
            sampler=lambda n: torch.rand(n, 2),
            value=lambda x: torch.full((x.shape[0], 1), target),
            n_pts=n_pts,
        )

    def test_loss_is_scalar(self):
        constraint = DirichletBCConstraint(self._make_bc(), self._make_field())
        loss = constraint.loss(DUMMY_BATCH, nn.MSELoss())
        assert loss.shape == ()

    def test_loss_is_non_negative(self):
        constraint = DirichletBCConstraint(self._make_bc(), self._make_field())
        loss = constraint.loss(DUMMY_BATCH, nn.MSELoss())
        assert loss.item() >= 0.0

    def test_weight_scales_loss(self):
        field = self._make_field()
        bc = self._make_bc()
        c1 = DirichletBCConstraint(bc, field, weight=1.0)
        c2 = DirichletBCConstraint(bc, field, weight=2.0)
        # fix the sampler output so both use same boundary points
        fixed = torch.rand(20, 2)
        bc.sampler = lambda n: fixed
        l1 = c1.loss(DUMMY_BATCH, nn.MSELoss())
        l2 = c2.loss(DUMMY_BATCH, nn.MSELoss())
        assert l2.item() == pytest.approx(2.0 * l1.item(), rel=1e-5)

    def test_log_called(self):
        log_calls: list[str] = []

        def log_fn(name: str, value: Tensor, progress_bar: bool = False) -> None:
            log_calls.append(name)

        constraint = DirichletBCConstraint(
            self._make_bc(), self._make_field(), log_key="loss/test_dir"
        )
        constraint.loss(DUMMY_BATCH, nn.MSELoss(), log=log_fn)
        assert log_calls == ["loss/test_dir"]

    def test_zero_loss_when_field_matches_target(self):
        """A field that always returns the target value should give zero Dirichlet loss."""
        field = _ZeroField(MLPConfig(in_dim=2, out_dim=1, hidden_layers=[4], activation="tanh"))
        bc = BoundaryCondition(
            sampler=lambda n: torch.rand(n, 2),
            value=lambda x: torch.zeros(x.shape[0], 1),
            n_pts=10,
        )
        constraint = DirichletBCConstraint(bc, field)
        loss = constraint.loss(DUMMY_BATCH, nn.MSELoss())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


class TestNeumannBCConstraint:
    def _make_field(self) -> Field:
        return Field(MLPConfig(in_dim=2, out_dim=1, hidden_layers=[8], activation="tanh"))

    def _make_bc(self, n_pts: int = 20) -> BoundaryCondition:
        return BoundaryCondition(
            sampler=lambda n: torch.rand(n, 2),
            value=lambda x: torch.zeros(x.shape[0], 1),
            n_pts=n_pts,
        )

    def test_loss_is_scalar(self):
        constraint = NeumannBCConstraint(self._make_bc(), self._make_field(), normal_dim=0)
        loss = constraint.loss(DUMMY_BATCH, nn.MSELoss())
        assert loss.shape == ()

    def test_loss_is_non_negative(self):
        constraint = NeumannBCConstraint(self._make_bc(), self._make_field(), normal_dim=1)
        loss = constraint.loss(DUMMY_BATCH, nn.MSELoss())
        assert loss.item() >= 0.0

    def test_log_called(self):
        log_calls: list[str] = []

        def log_fn(name: str, value: Tensor, progress_bar: bool = False) -> None:
            log_calls.append(name)

        constraint = NeumannBCConstraint(
            self._make_bc(), self._make_field(), normal_dim=0, log_key="loss/test_neu"
        )
        constraint.loss(DUMMY_BATCH, nn.MSELoss(), log=log_fn)
        assert log_calls == ["loss/test_neu"]

    def test_gradient_flows_through_loss(self):
        """Loss must be differentiable w.r.t. field parameters."""
        field = self._make_field()
        constraint = NeumannBCConstraint(self._make_bc(), field, normal_dim=0)
        loss = constraint.loss(DUMMY_BATCH, nn.MSELoss())
        loss.backward()
        grads = [p.grad for p in field.parameters()]
        assert any(g is not None for g in grads)

    def test_normal_dim_selects_correct_derivative(self):
        """u(x, y) = x => du/dx = 1, du/dy = 0. Neumann on dim=0 with target=1 → near-zero loss."""
        field = _LinearField(MLPConfig(in_dim=2, out_dim=1, hidden_layers=[4], activation="tanh"))
        bc = BoundaryCondition(
            sampler=lambda n: torch.rand(n, 2).requires_grad_(False),
            value=lambda x: torch.ones(x.shape[0], 1),  # du/dx = 1
            n_pts=20,
        )
        constraint = NeumannBCConstraint(bc, field, normal_dim=0)
        loss = constraint.loss(DUMMY_BATCH, nn.MSELoss())
        assert loss.item() == pytest.approx(0.0, abs=1e-5)
