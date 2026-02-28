"""Burgers Equation 1D — nonlinear PDE inverse problem definition (CSV data)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from anypinn.catalog.burgers_1d import NU_KEY, TRUE_NU, U_KEY, Burgers1DDataModule
from anypinn.core import (
    Field,
    FieldsRegistry,
    MLPConfig,
    Parameter,
    ParamsRegistry,
    PINNHyperparameters,
    Problem,
    RandomFourierFeatures,
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


def burgers_residual(x: Tensor, fields: FieldsRegistry, params: ParamsRegistry) -> Tensor:
    """PDE residual: du/dt + u*du/dx - nu*d2u/dx2 = 0."""
    u = fields[U_KEY](x)
    nu = params[NU_KEY](x)
    du_dt = partial(u, x, dim=1, order=1)
    du_dx = partial(u, x, dim=0, order=1)
    d2u_dx2 = partial(u, x, dim=0, order=2)
    return du_dt + u * du_dx - nu * d2u_dx2


# ============================================================================
# Boundary / IC Samplers — TODO: update for your domain
# ============================================================================


def _left_boundary(n: int) -> Tensor:
    return torch.stack([torch.full((n,), -1.0), torch.rand(n)], dim=1)


def _right_boundary(n: int) -> Tensor:
    return torch.stack([torch.ones(n), torch.rand(n)], dim=1)


def _initial_condition(n: int) -> Tensor:
    return torch.stack([2 * torch.rand(n) - 1, torch.zeros(n)], dim=1)


def _zero(x: Tensor) -> Tensor:
    """Homogeneous Dirichlet BC. TODO: update if non-zero."""
    return torch.zeros(x.shape[0], 1)


def _ic_value(x: Tensor) -> Tensor:
    """IC: u(x,0) = -sin(pi*x). TODO: update for your IC."""
    return -torch.sin(math.pi * x[:, 0:1])


# ============================================================================
# Residual Scorer for Adaptive Collocation
# ============================================================================


class BurgersResidualScorer:
    """ResidualScorer protocol implementation for adaptive collocation."""

    def __init__(self, fields: FieldsRegistry, params: ParamsRegistry) -> None:
        self.fields = fields
        self.params = params

    def residual_score(self, x: Tensor) -> Tensor:
        device = next(iter(self.fields.values())).parameters().__next__().device
        x = x.detach().to(device).requires_grad_(True)
        with torch.enable_grad():
            res = burgers_residual(x, self.fields, self.params)
        # res may be (n, d) due to scalar param broadcasting; reduce to (n,)
        return res.detach().cpu().abs().mean(dim=-1)


# ============================================================================
# Predict Data Function
# ============================================================================


def predict_data(x_data: Tensor, fields: FieldsRegistry, _params: ParamsRegistry) -> Tensor:
    return fields[U_KEY](x_data).unsqueeze(1)


# ============================================================================
# Data Module Factory
# ============================================================================

validation: ValidationRegistry = {NU_KEY: lambda x: torch.full_like(x, TRUE_NU)}


def create_data_module(
    hp: PINNHyperparameters,
    fields: FieldsRegistry,
    params: ParamsRegistry,
) -> Burgers1DDataModule:
    scorer = BurgersResidualScorer(fields, params)
    return Burgers1DDataModule(
        hp=hp,
        true_nu=TRUE_NU,
        grid_size=GRID_SIZE,
        residual_scorer=scorer,
        validation=validation,
    )


# ============================================================================
# Problem Factory
# ============================================================================


def create_problem(hp: PINNHyperparameters) -> Problem:
    rff = RandomFourierFeatures(in_dim=2, num_features=128, scale=1.0, seed=42)
    field_u = Field(
        config=MLPConfig(
            in_dim=rff.out_dim,
            out_dim=1,
            hidden_layers=hp.fields_config.hidden_layers,
            activation=hp.fields_config.activation,
            output_activation=hp.fields_config.output_activation,
            encode=rff,
        )
    )
    param_nu = Parameter(
        config=ScalarConfig(init_value=hp.params_config.init_value),
    )

    fields = FieldsRegistry({U_KEY: field_u})
    params = ParamsRegistry({NU_KEY: param_nu})

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
        residual_fn=burgers_residual,
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
