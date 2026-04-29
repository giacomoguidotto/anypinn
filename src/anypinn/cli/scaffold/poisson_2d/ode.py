"""Poisson 2D — PDE problem definition."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from anypinn.catalog.poisson_2d import K_KEY, TRUE_K, U_KEY, Poisson2DDataModule
from anypinn.core import (
    # --- VARIANT: direction/inverse ---
    DataConstraint,
    # --- END VARIANT ---
    Field,
    FieldsRegistry,
    FourierEncoding,
    MLPConfig,
    # --- VARIANT: direction/inverse ---
    Parameter,
    # --- END VARIANT ---
    ParamsRegistry,
    PINNHyperparameters,
    Problem,
    # --- VARIANT: direction/inverse ---
    ScalarConfig,
    ValidationRegistry,
    # --- END VARIANT ---
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


# --- VARIANT: direction/forward ---
def poisson_residual_forward(x: Tensor, fields: FieldsRegistry, _params: ParamsRegistry) -> Tensor:
    """PDE residual: -nabla^2(u) - f = 0 (k=1 known)."""
    u = fields[U_KEY](x)
    return -laplacian(u, x) - source_fn(x)


# --- VARIANT: direction/inverse ---
def poisson_residual_inverse(x: Tensor, fields: FieldsRegistry, params: ParamsRegistry) -> Tensor:
    """PDE residual: -k * nabla^2(u) - f = 0 (k learned)."""
    u = fields[U_KEY](x)
    k = params[K_KEY](x)
    return -k * laplacian(u, x) - source_fn(x)


# --- END VARIANT ---

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


# --- VARIANT: direction/inverse ---
# ============================================================================
# Predict Data Function
# ============================================================================


def predict_data(x_data: Tensor, fields: FieldsRegistry, _params: ParamsRegistry) -> Tensor:
    return fields[U_KEY](x_data).unsqueeze(1)


# --- END VARIANT ---

# ============================================================================
# Data Module Factory
# ============================================================================

# --- VARIANT: direction/inverse ---
_validation: ValidationRegistry = {K_KEY: lambda x: torch.full_like(x, TRUE_K)}
# --- VARIANT: direction/forward ---
_validation = None
# --- END VARIANT ---


# --- VARIANT: source/synthetic ---
def create_data_module_synthetic(hp: PINNHyperparameters) -> Poisson2DDataModule:
    return Poisson2DDataModule(
        hp=hp,
        grid_size=GRID_SIZE,
        validation=_validation,
    )


# --- VARIANT: source/csv ---
def create_data_module_csv(hp: PINNHyperparameters) -> Poisson2DDataModule:
    return Poisson2DDataModule(
        hp=hp,
        grid_size=GRID_SIZE,
        validation=_validation,
    )


# --- END VARIANT ---

# ============================================================================
# Problem Factory
# ============================================================================


# --- VARIANT: direction/forward ---
def create_problem_forward(hp: PINNHyperparameters) -> Problem:
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
        residual_fn=poisson_residual_forward,
        log_key="loss/pde_residual",
        weight=1.0,
    )

    return Problem(
        constraints=[pde, *bcs],
        criterion=build_criterion(hp.criterion),
        fields=fields,
        params=params,
    )


# --- VARIANT: direction/inverse ---
def create_problem_inverse(hp: PINNHyperparameters) -> Problem:
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
    param_k = Parameter(
        config=ScalarConfig(init_value=hp.params_config.init_value),
    )

    fields = FieldsRegistry({U_KEY: field_u})
    params = ParamsRegistry({K_KEY: param_k})

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
        residual_fn=poisson_residual_inverse,
        log_key="loss/pde_residual",
        weight=1.0,
    )

    data = DataConstraint(
        fields=fields,
        params=params,
        predict_data=predict_data,
        weight=5.0,
    )

    return Problem(
        constraints=[pde, *bcs, data],
        criterion=build_criterion(hp.criterion),
        fields=fields,
        params=params,
    )


# --- END VARIANT ---
