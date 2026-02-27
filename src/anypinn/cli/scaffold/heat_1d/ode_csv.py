"""Heat Equation 1D — inverse PDE problem definition (CSV data)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from anypinn.catalog.heat_1d import ALPHA_KEY, TRUE_ALPHA, U_KEY, Heat1DDataModule
from anypinn.core import (
    Field,
    FieldsRegistry,
    FourierEncoding,
    MLPConfig,
    Parameter,
    ParamsRegistry,
    PINNHyperparameters,
    Problem,
    ScalarConfig,
    ValidationRegistry,
    build_criterion,
)
from anypinn.lib.diff import partial
from anypinn.problems import (
    BoundaryCondition,
    DataConstraint,
    DirichletBCConstraint,
    PDEResidualConstraint,
)

# ============================================================================
# Constants
# ============================================================================

GRID_SIZE = 50

# ============================================================================
# PDE Definition
# ============================================================================


def heat_residual(x: Tensor, fields: FieldsRegistry, params: ParamsRegistry) -> Tensor:
    """PDE residual: du/dt - alpha * d2u/dx2 = 0."""
    u = fields[U_KEY](x)
    alpha = params[ALPHA_KEY](x)
    du_dt = partial(u, x, dim=1, order=1)
    d2u_dx2 = partial(u, x, dim=0, order=2)
    return du_dt - alpha * d2u_dx2


# ============================================================================
# Boundary / IC Samplers — TODO: update for your domain
# ============================================================================


def _left_boundary(n: int) -> Tensor:
    return torch.stack([torch.zeros(n), torch.rand(n)], dim=1)


def _right_boundary(n: int) -> Tensor:
    return torch.stack([torch.ones(n), torch.rand(n)], dim=1)


def _initial_condition(n: int) -> Tensor:
    return torch.stack([torch.rand(n), torch.zeros(n)], dim=1)


def _zero(x: Tensor) -> Tensor:
    """Homogeneous Dirichlet BC. TODO: update if non-zero."""
    return torch.zeros(x.shape[0], 1)


def _ic_value(x: Tensor) -> Tensor:
    """IC: u(x,0) = sin(pi*x). TODO: update for your IC."""
    return torch.sin(math.pi * x[:, 0:1])


# ============================================================================
# Predict Data Function
# ============================================================================


def predict_data(x_data: Tensor, fields: FieldsRegistry, _params: ParamsRegistry) -> Tensor:
    return fields[U_KEY](x_data).unsqueeze(1)


# ============================================================================
# Data Module Factory
# ============================================================================

validation: ValidationRegistry = {ALPHA_KEY: lambda x: torch.full_like(x, TRUE_ALPHA)}


def create_data_module(hp: PINNHyperparameters) -> Heat1DDataModule:
    return Heat1DDataModule(
        hp=hp,
        true_alpha=TRUE_ALPHA,
        grid_size=GRID_SIZE,
        validation=validation,
    )


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
    param_alpha = Parameter(
        config=ScalarConfig(init_value=hp.params_config.init_value),
    )

    fields = FieldsRegistry({U_KEY: field_u})
    params = ParamsRegistry({ALPHA_KEY: param_alpha})

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
        residual_fn=heat_residual,
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
