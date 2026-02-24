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
        x_bc = self.bc.sampler(self.bc.n_pts)
        u_pred = self.field(x_bc)
        g = self.bc.value(x_bc)
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
        x_bc = self.bc.sampler(self.bc.n_pts).detach().requires_grad_(True)
        u_pred = self.field(x_bc)
        du_dn = diff_partial(u_pred, x_bc, dim=self.normal_dim)
        h = self.bc.value(x_bc.detach())
        loss: Tensor = self.weight * criterion(du_dn, h)
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
