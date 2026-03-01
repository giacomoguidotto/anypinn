"""Boundary condition constraints for PDE problems."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias, override

import torch
from torch import Tensor
import torch.nn as nn

from anypinn.core import Constraint, Field, FieldsRegistry, LogFn, ParamsRegistry, TrainingBatch
from anypinn.lib.diff import partial as diff_partial

BCValueFn: TypeAlias = Callable[[Tensor], Tensor]
"""A callable that maps boundary coordinates (n_pts, d) → target values (n_pts, out_dim)."""

PDEResidualFn: TypeAlias = Callable[[Tensor, FieldsRegistry, ParamsRegistry], Tensor]
"""A callable (x, fields, params) → residual tensor, expected to be zero at the solution."""


class BoundaryCondition:
    """
    Pairs a boundary region sampler with a prescribed value function.

    Args:
        sampler: Callable ``(n_pts: int) -> Tensor`` of shape ``(n_pts, d)``.
            Called each training step to produce fresh boundary sample points.
        value: Callable ``Tensor -> Tensor`` giving the target value or normal
            derivative at boundary coordinates.
        n_pts: Number of boundary points sampled per training step.
    """

    def __init__(
        self,
        sampler: Callable[[int], Tensor],
        value: BCValueFn,
        n_pts: int = 100,
    ):
        self.sampler = sampler
        self.value = value
        self.n_pts = n_pts


class DirichletBCConstraint(Constraint):
    """
    Enforces the Dirichlet boundary condition: u(x_bc) = g(x_bc).
    Minimizes ``weight * criterion(u(x_bc), g(x_bc))``.

    Args:
        bc: Boundary condition (sampler + target value function).
        field: The neural field to enforce the condition on.
        log_key: Key used when logging the loss value.
        weight: Loss term weight.
    """

    def __init__(
        self,
        bc: BoundaryCondition,
        field: Field,
        log_key: str = "loss/bc_dirichlet",
        weight: float = 1.0,
    ):
        self.bc = bc
        self.field = field
        self.log_key = log_key
        self.weight = weight

    @override
    def loss(
        self,
        batch: TrainingBatch,
        criterion: nn.Module,
        log: LogFn | None = None,
    ) -> Tensor:
        device = next(self.field.parameters()).device
        x_bc = self.bc.sampler(self.bc.n_pts).to(device)
        u_pred = self.field(x_bc)
        g = self.bc.value(x_bc).to(device)
        loss: Tensor = self.weight * criterion(u_pred, g)
        if log is not None:
            log(self.log_key, loss)
        return loss


class NeumannBCConstraint(Constraint):
    """
    Enforces the Neumann boundary condition:
    $\\partial u / \\partial n (x_{bc}) = h(x_{bc})$.

    For a rectangular domain face whose outward normal is axis-aligned with
    dimension ``normal_dim``, we have
    $\\partial u / \\partial n = \\partial u / \\partial x_{\\mathrm{normal\\_dim}}$.
    Minimizes
    ``weight * criterion(du_dn(x_bc), h(x_bc))``.

    Args:
        bc: Boundary condition (sampler + target normal-derivative function).
        field: The neural field to enforce the condition on.
        normal_dim: Index of the spatial dimension the boundary normal points along.
        log_key: Key used when logging the loss value.
        weight: Loss term weight.
    """

    def __init__(
        self,
        bc: BoundaryCondition,
        field: Field,
        normal_dim: int,
        log_key: str = "loss/bc_neumann",
        weight: float = 1.0,
    ):
        self.bc = bc
        self.field = field
        self.normal_dim = normal_dim
        self.log_key = log_key
        self.weight = weight

    @override
    def loss(
        self,
        batch: TrainingBatch,
        criterion: nn.Module,
        log: LogFn | None = None,
    ) -> Tensor:
        device = next(self.field.parameters()).device
        x_bc = self.bc.sampler(self.bc.n_pts).to(device).detach().requires_grad_(True)
        u_pred = self.field(x_bc)
        du_dn = diff_partial(u_pred, x_bc, dim=self.normal_dim)
        h = self.bc.value(x_bc.detach()).to(device)
        loss: Tensor = self.weight * criterion(du_dn, h)
        if log is not None:
            log(self.log_key, loss)
        return loss


class PeriodicBCConstraint(Constraint):
    """
    Enforces periodic boundary conditions:
    ``u(x_left, t) = u(x_right, t)`` and
    ``∂u/∂x(x_left, t) = ∂u/∂x(x_right, t)``.

    The two boundary samplers must produce **paired** points — identical
    coordinates in every dimension except the periodic one — so that
    the value- and derivative-matching losses are meaningful.

    Minimizes
    ``weight * [criterion(u_left, u_right) + criterion(du_left, du_right)]``.

    Args:
        bc_left: Left boundary sampler (sampler + dummy value function).
        bc_right: Right boundary sampler (sampler + dummy value function).
        field: The neural field to enforce the condition on.
        match_dim: Spatial dimension index for the derivative matching.
        log_key: Key used when logging the loss value.
        weight: Loss term weight.
    """

    def __init__(
        self,
        bc_left: BoundaryCondition,
        bc_right: BoundaryCondition,
        field: Field,
        match_dim: int = 0,
        log_key: str = "loss/bc_periodic",
        weight: float = 1.0,
    ):
        self.bc_left = bc_left
        self.bc_right = bc_right
        self.field = field
        self.match_dim = match_dim
        self.log_key = log_key
        self.weight = weight

    @override
    def loss(
        self,
        batch: TrainingBatch,
        criterion: nn.Module,
        log: LogFn | None = None,
    ) -> Tensor:
        device = next(self.field.parameters()).device
        n_pts = self.bc_left.n_pts

        x_left = self.bc_left.sampler(n_pts).to(device).detach().requires_grad_(True)
        x_right = self.bc_right.sampler(n_pts).to(device).detach().requires_grad_(True)

        u_left = self.field(x_left)
        u_right = self.field(x_right)

        # Value matching: u(x_left, t) = u(x_right, t)
        loss_val: Tensor = criterion(u_left, u_right)

        # Derivative matching: du/dx(x_left, t) = du/dx(x_right, t)
        du_left = diff_partial(u_left, x_left, dim=self.match_dim)
        du_right = diff_partial(u_right, x_right, dim=self.match_dim)
        loss_deriv: Tensor = criterion(du_left, du_right)

        loss: Tensor = self.weight * (loss_val + loss_deriv)
        if log is not None:
            log(self.log_key, loss)
        return loss


class PDEResidualConstraint(Constraint):
    """
    Enforces a PDE interior residual: ``residual_fn(x, fields, params) ≈ 0``.
    Minimizes ``weight * criterion(residual_fn(x_coll, fields, params), 0)``.

    Args:
        fields: Registry of neural fields the residual function operates on.
            Pass only the subset needed — other fields in the Problem are ignored.
        params: Registry of parameters the residual function uses.
        residual_fn: Callable (x, fields, params) → Tensor of residuals.
            Should use ``anypinn.lib.diff`` operators for derivatives.
            The returned tensor is compared against zeros.
        log_key: Key used when logging the loss value.
        weight: Loss term weight.
    """

    def __init__(
        self,
        fields: FieldsRegistry,
        params: ParamsRegistry,
        residual_fn: PDEResidualFn,
        log_key: str = "loss/pde_residual",
        weight: float = 1.0,
    ):
        self.fields = fields
        self.params = params
        self.residual_fn = residual_fn
        self.log_key = log_key
        self.weight = weight

    @override
    def loss(
        self,
        batch: TrainingBatch,
        criterion: nn.Module,
        log: LogFn | None = None,
    ) -> Tensor:
        _, x_coll = batch
        x_coll = x_coll.detach().requires_grad_(True)
        residual = self.residual_fn(x_coll, self.fields, self.params)
        loss: Tensor = self.weight * criterion(residual, torch.zeros_like(residual))
        if log is not None:
            log(self.log_key, loss)
        return loss
