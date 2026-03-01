"""Gray-Scott 2D Reaction-Diffusion — inverse PDE problem definition (CSV data)."""

from __future__ import annotations

import torch
from torch import Tensor

from anypinn.catalog.gray_scott_2d import (
    DU_KEY,
    DV_KEY,
    F_KEY,
    K_KEY,
    T_TOTAL,
    TRUE_DU,
    TRUE_DV,
    TRUE_F,
    TRUE_K,
    U_KEY,
    V_KEY,
    GrayScott2DDataModule,
)
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
# PDE Definition
# ============================================================================


def gs_residual_u(x: Tensor, fields: FieldsRegistry, params: ParamsRegistry) -> Tensor:
    """PDE residual for u: du/dt_norm - T * (D_u lap_u - uv^2 + F(1-u)) = 0."""
    u = fields[U_KEY](x)
    v = fields[V_KEY](x)
    du = params[DU_KEY](x)
    f = params[F_KEY](x)
    du_dt = partial(u, x, dim=2, order=1)
    lap_u = partial(u, x, dim=0, order=2) + partial(u, x, dim=1, order=2)
    return du_dt - (du * lap_u - u * v**2 + f * (1 - u)) * T_TOTAL


def gs_residual_v(x: Tensor, fields: FieldsRegistry, params: ParamsRegistry) -> Tensor:
    """PDE residual for v: dv/dt_norm - T * (D_v lap_v + uv^2 - (F+k)v) = 0."""
    u = fields[U_KEY](x)
    v = fields[V_KEY](x)
    dv = params[DV_KEY](x)
    f = params[F_KEY](x)
    k = params[K_KEY](x)
    dv_dt = partial(v, x, dim=2, order=1)
    lap_v = partial(v, x, dim=0, order=2) + partial(v, x, dim=1, order=2)
    return dv_dt - (dv * lap_v + u * v**2 - (f + k) * v) * T_TOTAL


# ============================================================================
# Boundary / IC Samplers — TODO: update for your domain
# ============================================================================


def _left_edge(n: int) -> Tensor:
    return torch.stack([torch.zeros(n), torch.rand(n), torch.rand(n)], dim=1)


def _right_edge(n: int) -> Tensor:
    return torch.stack([torch.ones(n), torch.rand(n), torch.rand(n)], dim=1)


def _bottom_edge(n: int) -> Tensor:
    return torch.stack([torch.rand(n), torch.zeros(n), torch.rand(n)], dim=1)


def _top_edge(n: int) -> Tensor:
    return torch.stack([torch.rand(n), torch.ones(n), torch.rand(n)], dim=1)


def _initial_condition(n: int) -> Tensor:
    return torch.stack([torch.rand(n), torch.rand(n), torch.zeros(n)], dim=1)


def _zero(x: Tensor) -> Tensor:
    """Homogeneous Neumann BC. TODO: update if non-zero."""
    return torch.zeros(x.shape[0], 1)


def _ic_u(x: Tensor) -> Tensor:
    """IC for u. TODO: update for your IC."""
    vals = torch.ones(x.shape[0], 1)
    center = (x[:, 0] >= 0.4) & (x[:, 0] <= 0.6) & (x[:, 1] >= 0.4) & (x[:, 1] <= 0.6)
    vals[center] = 0.5
    return vals


def _ic_v(x: Tensor) -> Tensor:
    """IC for v. TODO: update for your IC."""
    vals = torch.zeros(x.shape[0], 1)
    center = (x[:, 0] >= 0.4) & (x[:, 0] <= 0.6) & (x[:, 1] >= 0.4) & (x[:, 1] <= 0.6)
    vals[center] = 0.25
    return vals


# ============================================================================
# Predict Data Function
# ============================================================================


def predict_data(x_data: Tensor, fields: FieldsRegistry, _params: ParamsRegistry) -> Tensor:
    u_pred = fields[U_KEY](x_data)
    v_pred = fields[V_KEY](x_data)
    return torch.stack([u_pred, v_pred], dim=1)


# ============================================================================
# Data Module Factory
# ============================================================================

validation: ValidationRegistry = {
    DU_KEY: lambda x: torch.full_like(x, TRUE_DU),
    DV_KEY: lambda x: torch.full_like(x, TRUE_DV),
    F_KEY: lambda x: torch.full_like(x, TRUE_F),
    K_KEY: lambda x: torch.full_like(x, TRUE_K),
}


def create_data_module(hp: PINNHyperparameters) -> GrayScott2DDataModule:
    return GrayScott2DDataModule(
        hp=hp,
        true_du=TRUE_DU,
        true_dv=TRUE_DV,
        true_f=TRUE_F,
        true_k=TRUE_K,
        validation=validation,
    )


# ============================================================================
# Problem Factory
# ============================================================================


def create_problem(hp: PINNHyperparameters) -> Problem:
    encode = FourierEncoding(num_frequencies=6)
    field_u = Field(
        config=MLPConfig(
            in_dim=encode.out_dim(3),
            out_dim=1,
            hidden_layers=hp.fields_config.hidden_layers,
            activation=hp.fields_config.activation,
            output_activation=hp.fields_config.output_activation,
            encode=encode,
        )
    )
    field_v = Field(
        config=MLPConfig(
            in_dim=encode.out_dim(3),
            out_dim=1,
            hidden_layers=hp.fields_config.hidden_layers,
            activation=hp.fields_config.activation,
            output_activation=hp.fields_config.output_activation,
            encode=encode,
        )
    )
    param_du = Parameter(config=ScalarConfig(init_value=hp.params_config.init_value))
    param_dv = Parameter(config=ScalarConfig(init_value=hp.params_config.init_value))
    param_f = Parameter(config=ScalarConfig(init_value=hp.params_config.init_value))
    param_k = Parameter(config=ScalarConfig(init_value=hp.params_config.init_value))

    fields = FieldsRegistry({U_KEY: field_u, V_KEY: field_v})
    params = ParamsRegistry(
        {
            DU_KEY: param_du,
            DV_KEY: param_dv,
            F_KEY: param_f,
            K_KEY: param_k,
        }
    )

    bcs = [
        NeumannBCConstraint(
            BoundaryCondition(sampler=_left_edge, value=_zero, n_pts=100),
            field_u,
            normal_dim=0,
            log_key="loss/bc_left_u",
            weight=10.0,
        ),
        NeumannBCConstraint(
            BoundaryCondition(sampler=_left_edge, value=_zero, n_pts=100),
            field_v,
            normal_dim=0,
            log_key="loss/bc_left_v",
            weight=10.0,
        ),
        NeumannBCConstraint(
            BoundaryCondition(sampler=_right_edge, value=_zero, n_pts=100),
            field_u,
            normal_dim=0,
            log_key="loss/bc_right_u",
            weight=10.0,
        ),
        NeumannBCConstraint(
            BoundaryCondition(sampler=_right_edge, value=_zero, n_pts=100),
            field_v,
            normal_dim=0,
            log_key="loss/bc_right_v",
            weight=10.0,
        ),
        NeumannBCConstraint(
            BoundaryCondition(sampler=_bottom_edge, value=_zero, n_pts=100),
            field_u,
            normal_dim=1,
            log_key="loss/bc_bottom_u",
            weight=10.0,
        ),
        NeumannBCConstraint(
            BoundaryCondition(sampler=_bottom_edge, value=_zero, n_pts=100),
            field_v,
            normal_dim=1,
            log_key="loss/bc_bottom_v",
            weight=10.0,
        ),
        NeumannBCConstraint(
            BoundaryCondition(sampler=_top_edge, value=_zero, n_pts=100),
            field_u,
            normal_dim=1,
            log_key="loss/bc_top_u",
            weight=10.0,
        ),
        NeumannBCConstraint(
            BoundaryCondition(sampler=_top_edge, value=_zero, n_pts=100),
            field_v,
            normal_dim=1,
            log_key="loss/bc_top_v",
            weight=10.0,
        ),
        DirichletBCConstraint(
            BoundaryCondition(sampler=_initial_condition, value=_ic_u, n_pts=200),
            field_u,
            log_key="loss/ic_u",
            weight=10.0,
        ),
        DirichletBCConstraint(
            BoundaryCondition(sampler=_initial_condition, value=_ic_v, n_pts=200),
            field_v,
            log_key="loss/ic_v",
            weight=10.0,
        ),
    ]

    pde_u = PDEResidualConstraint(
        fields=fields,
        params=params,
        residual_fn=gs_residual_u,
        log_key="loss/pde_u",
        weight=1.0,
    )
    pde_v = PDEResidualConstraint(
        fields=fields,
        params=params,
        residual_fn=gs_residual_v,
        log_key="loss/pde_v",
        weight=1.0,
    )

    data = DataConstraint(
        fields=fields,
        params=params,
        predict_data=predict_data,
        weight=5.0,
    )

    return Problem(
        constraints=[pde_u, pde_v, *bcs, data],
        criterion=build_criterion(hp.criterion),
        fields=fields,
        params=params,
    )
