"""Poisson 2D — PDE problem definition (synthetic)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from anypinn.catalog.poisson_2d import U_KEY, Poisson2DDataModule
from anypinn.core import (
    Field,
    FieldsRegistry,
    FourierEncoding,
    MLPConfig,
    ParamsRegistry,
    PINNHyperparameters,
    Problem,
    build_criterion,
)
from anypinn.lib.diff import laplacian
from anypinn.problems import BoundaryCondition, DirichletBCConstraint, PDEResidualConstraint

# ============================================================================
# Constants
# ============================================================================

GRID_SIZE = 30

# ============================================================================
# PDE Definition
# ============================================================================


def source_fn(x: Tensor) -> Tensor:
    """Source term f(x,y) = -2 pi^2 sin(pi*x) sin(pi*y)."""
    return -2 * math.pi**2 * torch.sin(math.pi * x[:, 0:1]) * torch.sin(math.pi * x[:, 1:2])


def poisson_residual(x: Tensor, fields: FieldsRegistry, params: ParamsRegistry) -> Tensor:
    """PDE residual: laplacian(u) - f = 0."""
    u = fields[U_KEY](x)
    return laplacian(u, x) - source_fn(x)


# ============================================================================
# Boundary Samplers
# ============================================================================


def _left_edge(n: int) -> Tensor:
    return torch.stack([torch.zeros(n), torch.rand(n)], dim=1)


def _right_edge(n: int) -> Tensor:
    return torch.stack([torch.ones(n), torch.rand(n)], dim=1)


def _bottom_edge(n: int) -> Tensor:
    return torch.stack([torch.rand(n), torch.zeros(n)], dim=1)


def _top_edge(n: int) -> Tensor:
    return torch.stack([torch.rand(n), torch.ones(n)], dim=1)


def _zero(x: Tensor) -> Tensor:
    return torch.zeros(x.shape[0], 1)


# ============================================================================
# Data Module Factory
# ============================================================================


def create_data_module(hp: PINNHyperparameters) -> Poisson2DDataModule:
    return Poisson2DDataModule(hp=hp, grid_size=GRID_SIZE)


# ============================================================================
# Problem Factory
# ============================================================================


def create_problem(hp: PINNHyperparameters) -> Problem:
    encode = FourierEncoding(num_frequencies=6)
    field_u = Field(
        config=MLPConfig(
            in_dim=encode.out_dim(2),
            out_dim=1,
            hidden_layers=hp.fields_config.hidden_layers,
            activation=hp.fields_config.activation,
            output_activation=hp.fields_config.output_activation,
            encode=encode,
        )
    )
    fields = FieldsRegistry({U_KEY: field_u})
    params = ParamsRegistry({})

    bcs = [
        DirichletBCConstraint(
            BoundaryCondition(sampler=_left_edge, value=_zero, n_pts=100),
            field_u,
            log_key="loss/bc_left",
            weight=10.0,
        ),
        DirichletBCConstraint(
            BoundaryCondition(sampler=_right_edge, value=_zero, n_pts=100),
            field_u,
            log_key="loss/bc_right",
            weight=10.0,
        ),
        DirichletBCConstraint(
            BoundaryCondition(sampler=_bottom_edge, value=_zero, n_pts=100),
            field_u,
            log_key="loss/bc_bottom",
            weight=10.0,
        ),
        DirichletBCConstraint(
            BoundaryCondition(sampler=_top_edge, value=_zero, n_pts=100),
            field_u,
            log_key="loss/bc_top",
            weight=10.0,
        ),
    ]

    pde = PDEResidualConstraint(
        fields=fields,
        params=params,
        residual_fn=poisson_residual,
        log_key="loss/pde_residual",
        weight=1.0,
    )

    return Problem(
        constraints=[pde, *bcs],
        criterion=build_criterion(hp.criterion),
        fields=fields,
        params=params,
    )
