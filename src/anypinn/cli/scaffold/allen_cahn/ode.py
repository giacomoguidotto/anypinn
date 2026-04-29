"""Allen-Cahn Equation — stiff reaction-diffusion problem definition."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from anypinn.catalog.allen_cahn import EPSILON_KEY, TRUE_EPSILON, U_KEY, AllenCahnDataModule
from anypinn.core import (
    # --- VARIANT: direction/inverse ---
    DataConstraint,
    # --- END VARIANT ---
    Field,
    FieldsRegistry,
    MLPConfig,
    # --- VARIANT: direction/inverse ---
    Parameter,
    # --- END VARIANT ---
    ParamsRegistry,
    PINNHyperparameters,
    Problem,
    RandomFourierFeatures,
    # --- VARIANT: direction/inverse ---
    ScalarConfig,
    ValidationRegistry,
    # --- END VARIANT ---
    build_criterion,
)
from anypinn.lib.diff import partial
from anypinn.problems import (
    BoundaryCondition,
    DirichletBCConstraint,
    PDEResidualConstraint,
    PeriodicBCConstraint,
)

# ============================================================================
# Constants
# ============================================================================

GRID_SIZE = 50
EPSILON = TRUE_EPSILON

# ============================================================================
# PDE Definition
# ============================================================================


# --- VARIANT: direction/forward ---
def allen_cahn_residual_forward(
    x: Tensor, fields: FieldsRegistry, _params: ParamsRegistry
) -> Tensor:
    """PDE residual: du/dt - eps*d2u/dx2 - u + u^3 = 0 (eps known)."""
    u = fields[U_KEY](x)
    du_dt = partial(u, x, dim=1, order=1)
    d2u_dx2 = partial(u, x, dim=0, order=2)
    return du_dt - EPSILON * d2u_dx2 - u + u**3


# --- VARIANT: direction/inverse ---
def allen_cahn_residual_inverse(
    x: Tensor, fields: FieldsRegistry, params: ParamsRegistry
) -> Tensor:
    """PDE residual: du/dt - eps*d2u/dx2 - u + u^3 = 0 (eps learned)."""
    u = fields[U_KEY](x)
    eps = params[EPSILON_KEY](x)
    du_dt = partial(u, x, dim=1, order=1)
    d2u_dx2 = partial(u, x, dim=0, order=2)
    return du_dt - eps * d2u_dx2 - u + u**3


# --- END VARIANT ---

# ============================================================================
# Boundary / IC Samplers
# ============================================================================


def _periodic_left(n: int) -> Tensor:
    t = torch.rand(n)
    return torch.stack([torch.full((n,), -1.0), t], dim=1)


def _periodic_right(n: int) -> Tensor:
    t = torch.rand(n)
    return torch.stack([torch.ones(n), t], dim=1)


def _initial_condition(n: int) -> Tensor:
    return torch.stack([2 * torch.rand(n) - 1, torch.zeros(n)], dim=1)


def _dummy(x: Tensor) -> Tensor:
    return torch.zeros(x.shape[0], 1)


def _ic_value(x: Tensor) -> Tensor:
    scale = math.sqrt(2 * EPSILON)
    return -torch.tanh(x[:, 0:1] / scale)


# ============================================================================
# Residual Scorer for Adaptive Collocation
# ============================================================================


class AllenCahnResidualScorer:
    """ResidualScorer protocol implementation for adaptive collocation."""

    def __init__(self, fields: FieldsRegistry, params: ParamsRegistry) -> None:
        self.fields = fields
        self.params = params

    def residual_score(self, x: Tensor) -> Tensor:
        device = next(iter(self.fields.values())).parameters().__next__().device
        x = x.detach().to(device).requires_grad_(True)
        with torch.enable_grad():
            res = allen_cahn_residual_forward(x, self.fields, self.params)
        return res.detach().cpu().abs().mean(dim=-1)


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
_validation: ValidationRegistry = {EPSILON_KEY: lambda x: torch.full_like(x, TRUE_EPSILON)}
# --- VARIANT: direction/forward ---
_validation = None
# --- END VARIANT ---


# --- VARIANT: source/synthetic ---
def create_data_module_synthetic(hp: PINNHyperparameters) -> AllenCahnDataModule:
    # To enable adaptive collocation, pass residual_scorer=AllenCahnResidualScorer(fields, params)
    return AllenCahnDataModule(
        hp=hp,
        true_epsilon=TRUE_EPSILON,
        grid_size=GRID_SIZE,
        validation=_validation,
    )


# --- VARIANT: source/csv ---
def create_data_module_csv(hp: PINNHyperparameters) -> AllenCahnDataModule:
    # To enable adaptive collocation, pass residual_scorer=AllenCahnResidualScorer(fields, params)
    return AllenCahnDataModule(
        hp=hp,
        true_epsilon=TRUE_EPSILON,
        grid_size=GRID_SIZE,
        validation=_validation,
    )


# --- END VARIANT ---

# ============================================================================
# Problem Factory
# ============================================================================


# --- VARIANT: direction/forward ---
def create_problem_forward(hp: PINNHyperparameters) -> Problem:
    rff = RandomFourierFeatures(in_dim=2, num_features=128, scale=5.0, seed=42)
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

    fields = FieldsRegistry({U_KEY: field_u})
    params = ParamsRegistry({})

    bcs = [
        PeriodicBCConstraint(
            bc_left=BoundaryCondition(sampler=_periodic_left, value=_dummy, n_pts=100),
            bc_right=BoundaryCondition(sampler=_periodic_right, value=_dummy, n_pts=100),
            field=field_u,
            match_dim=0,
            log_key="loss/bc_periodic",
            weight=10.0,
        ),
        DirichletBCConstraint(
            BoundaryCondition(sampler=_initial_condition, value=_ic_value, n_pts=200),
            field_u,
            log_key="loss/ic",
            weight=10.0,
        ),
    ]

    pde = PDEResidualConstraint(
        fields=fields,
        params=params,
        residual_fn=allen_cahn_residual_forward,
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
    rff = RandomFourierFeatures(in_dim=2, num_features=128, scale=5.0, seed=42)
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
    param_epsilon = Parameter(
        config=ScalarConfig(init_value=hp.params_config.init_value),
    )

    fields = FieldsRegistry({U_KEY: field_u})
    params = ParamsRegistry({EPSILON_KEY: param_epsilon})

    bcs = [
        PeriodicBCConstraint(
            bc_left=BoundaryCondition(sampler=_periodic_left, value=_dummy, n_pts=100),
            bc_right=BoundaryCondition(sampler=_periodic_right, value=_dummy, n_pts=100),
            field=field_u,
            match_dim=0,
            log_key="loss/bc_periodic",
            weight=10.0,
        ),
        DirichletBCConstraint(
            BoundaryCondition(sampler=_initial_condition, value=_ic_value, n_pts=200),
            field_u,
            log_key="loss/ic",
            weight=10.0,
        ),
    ]

    pde = PDEResidualConstraint(
        fields=fields,
        params=params,
        residual_fn=allen_cahn_residual_inverse,
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
