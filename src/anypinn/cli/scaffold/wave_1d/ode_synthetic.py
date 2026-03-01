"""Wave Equation 1D — inverse PDE problem definition (synthetic)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from anypinn.catalog.wave_1d import C_KEY, TRUE_C, U_KEY, Wave1DDataModule
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
    NeumannBCConstraint,
    PDEResidualConstraint,
)

# ============================================================================
# Constants
# ============================================================================

GRID_SIZE = 50

# ============================================================================
# PDE Definition
# ============================================================================


def wave_residual(x: Tensor, fields: FieldsRegistry, params: ParamsRegistry) -> Tensor:
    """PDE residual: d2u/dt2 - c^2 * d2u/dx2 = 0."""
    u = fields[U_KEY](x)
    c = params[C_KEY](x)
    d2u_dt2 = partial(u, x, dim=1, order=2)
    d2u_dx2 = partial(u, x, dim=0, order=2)
    return d2u_dt2 - c * c * d2u_dx2


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


# ============================================================================
# Predict Data Function
# ============================================================================


def predict_data(x_data: Tensor, fields: FieldsRegistry, _params: ParamsRegistry) -> Tensor:
    return fields[U_KEY](x_data).unsqueeze(1)


# ============================================================================
# Data Module Factory
# ============================================================================

validation: ValidationRegistry = {C_KEY: lambda x: torch.full_like(x, TRUE_C)}


def create_data_module(hp: PINNHyperparameters) -> Wave1DDataModule:
    return Wave1DDataModule(
        hp=hp,
        true_c=TRUE_C,
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
    param_c = Parameter(
        config=ScalarConfig(init_value=hp.params_config.init_value),
    )

    fields = FieldsRegistry({U_KEY: field_u})
    params = ParamsRegistry({C_KEY: param_c})

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
        NeumannBCConstraint(
            BoundaryCondition(sampler=_initial_condition, value=_zero, n_pts=100),
            field_u,
            normal_dim=1,
            log_key="loss/ic_velocity",
            weight=10.0,
        ),
    ]

    pde = PDEResidualConstraint(
        fields=fields,
        params=params,
        residual_fn=wave_residual,
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
