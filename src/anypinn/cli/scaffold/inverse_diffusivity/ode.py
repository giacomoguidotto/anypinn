"""Inverse Diffusivity — PDE problem definition."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from anypinn.catalog.inverse_diffusivity import (
    D_KEY,
    TRUE_D_FN,
    U_KEY,
    InverseDiffusivityDataModule,
)
from anypinn.core import (
    # --- VARIANT: direction/inverse ---
    DataConstraint,
    # --- END VARIANT ---
    Field,
    FieldsRegistry,
    FourierEncoding,
    MLPConfig,
    ParamsRegistry,
    PINNHyperparameters,
    Problem,
    ValidationRegistry,
    build_criterion,
)
from anypinn.lib.diff import partial
from anypinn.problems import BoundaryCondition, DirichletBCConstraint, PDEResidualConstraint

# ============================================================================
# Constants
# ============================================================================

GRID_SIZE = 50

# ============================================================================
# PDE Definition
# ============================================================================


# --- VARIANT: direction/forward ---
def diffusivity_residual_forward(
    x: Tensor, fields: FieldsRegistry, _params: ParamsRegistry
) -> Tensor:
    """PDE residual: du/dt - d/dx(D(x) du/dx) = 0 (D(x) known)."""
    u = fields[U_KEY](x)
    D = TRUE_D_FN(x[:, 0:1])
    dD_dx = 2 * math.pi * 0.05 * torch.cos(2 * math.pi * x[:, 0:1])
    du_dt = partial(u, x, dim=1, order=1)
    du_dx = partial(u, x, dim=0, order=1)
    d2u_dx2 = partial(u, x, dim=0, order=2)
    return du_dt - (dD_dx * du_dx + D * d2u_dx2)


# --- VARIANT: direction/inverse ---
def diffusivity_residual_inverse(
    x: Tensor, fields: FieldsRegistry, _params: ParamsRegistry
) -> Tensor:
    """PDE residual: du/dt - d/dx(D(x) du/dx) = 0 (D(x) learned)."""
    u = fields[U_KEY](x)
    D = fields[D_KEY](x)
    du_dt = partial(u, x, dim=1, order=1)
    du_dx = partial(u, x, dim=0, order=1)
    d2u_dx2 = partial(u, x, dim=0, order=2)
    dD_dx = partial(D, x, dim=0, order=1)
    return du_dt - (dD_dx * du_dx + D * d2u_dx2)


# --- END VARIANT ---

# ============================================================================
# Boundary / IC Samplers
# ============================================================================


def _left_boundary(n: int) -> Tensor:
    return torch.stack([torch.zeros(n), torch.rand(n)], dim=1)


def _right_boundary(n: int) -> Tensor:
    return torch.stack([torch.ones(n), torch.rand(n)], dim=1)


def _initial_condition(n: int) -> Tensor:
    return torch.stack([torch.rand(n), torch.zeros(n)], dim=1)


def _zero(x: Tensor) -> Tensor:
    return torch.zeros(x.shape[0], 1)


def _ic_value(x: Tensor) -> Tensor:
    return torch.sin(math.pi * x[:, 0:1])


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
_validation: ValidationRegistry = {D_KEY: lambda x: TRUE_D_FN(x[:, 0:1])}
# --- VARIANT: direction/forward ---
_validation = None
# --- END VARIANT ---


# --- VARIANT: source/synthetic ---
def create_data_module_synthetic(hp: PINNHyperparameters) -> InverseDiffusivityDataModule:
    return InverseDiffusivityDataModule(
        hp=hp,
        grid_size=GRID_SIZE,
        validation=_validation,
    )


# --- VARIANT: source/csv ---
def create_data_module_csv(hp: PINNHyperparameters) -> InverseDiffusivityDataModule:
    return InverseDiffusivityDataModule(
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
            BoundaryCondition(sampler=_left_boundary, value=_zero, n_pts=100),
            field_u,
            log_key="loss/bc_left",
            weight=10.0,
        ),
        DirichletBCConstraint(
            BoundaryCondition(sampler=_right_boundary, value=_zero, n_pts=100),
            field_u,
            log_key="loss/bc_right",
            weight=10.0,
        ),
        DirichletBCConstraint(
            BoundaryCondition(sampler=_initial_condition, value=_ic_value, n_pts=100),
            field_u,
            log_key="loss/ic",
            weight=10.0,
        ),
    ]

    pde = PDEResidualConstraint(
        fields=fields,
        params=params,
        residual_fn=diffusivity_residual_forward,
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
    field_d = Field(
        config=MLPConfig(
            in_dim=encode.out_dim(2),
            out_dim=1,
            hidden_layers=hp.fields_config.hidden_layers,
            activation=hp.fields_config.activation,
            output_activation="softplus",
            encode=encode,
        )
    )

    fields = FieldsRegistry({U_KEY: field_u, D_KEY: field_d})
    params = ParamsRegistry({})

    bcs = [
        DirichletBCConstraint(
            BoundaryCondition(sampler=_left_boundary, value=_zero, n_pts=100),
            field_u,
            log_key="loss/bc_left",
            weight=10.0,
        ),
        DirichletBCConstraint(
            BoundaryCondition(sampler=_right_boundary, value=_zero, n_pts=100),
            field_u,
            log_key="loss/bc_right",
            weight=10.0,
        ),
        DirichletBCConstraint(
            BoundaryCondition(sampler=_initial_condition, value=_ic_value, n_pts=100),
            field_u,
            log_key="loss/ic",
            weight=10.0,
        ),
    ]

    pde = PDEResidualConstraint(
        fields=fields,
        params=params,
        residual_fn=diffusivity_residual_inverse,
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
