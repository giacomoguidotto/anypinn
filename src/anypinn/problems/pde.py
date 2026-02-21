"""Boundary condition constraints for PDE problems."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias, override

import torch
from torch import Tensor
import torch.nn as nn

from anypinn.core import Constraint, Field, LogFn, TrainingBatch

BCValueFn: TypeAlias = Callable[[Tensor], Tensor]
"""A callable that maps boundary coordinates (n_pts, d) → target values (n_pts, out_dim)."""


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
    Enforces the Neumann boundary condition: du/dn(x_bc) = h(x_bc).

    For a rectangular domain face whose outward normal is axis-aligned with
    dimension ``normal_dim``, ``du/dn = ∂u/∂x_{normal_dim}``.
    Minimizes ``weight * criterion(du/dn(x_bc), h(x_bc))``.

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
        # Full spatial Jacobian, project onto the outward normal dimension
        grad_u = torch.autograd.grad(u_pred.sum(), x_bc, create_graph=True)[0]  # (n_pts, d)
        du_dn = grad_u[:, self.normal_dim : self.normal_dim + 1]  # (n_pts, 1)
        h = self.bc.value(x_bc.detach())
        loss: Tensor = self.weight * criterion(du_dn, h)
        if log is not None:
            log(self.log_key, loss)
        return loss
