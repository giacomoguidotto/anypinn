"""Gray-Scott 2D Reaction-Diffusion — inverse PDE problem definition.

Solves the coupled system on [0,1]^2 x [0, T_TOTAL]:
  du/dt = D_u nabla^2 u - u v^2 + F(1-u)
  dv/dt = D_v nabla^2 v + u v^2 - (F+k) v

BCs: Neumann zero-flux on all 4 edges
ICs: u=1, v=0 everywhere; center square u=0.5, v=0.25

Inverse problem: recover D_u, D_v, F, k from sparse snapshots of u and v.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    Predictions,
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
# Boundary / IC Samplers — 3D coordinates (x, y, t_norm)
# ============================================================================


def _left_edge(n: int) -> Tensor:
    """x=0, y ~ U(0,1), t ~ U(0,1)."""
    return torch.stack([torch.zeros(n), torch.rand(n), torch.rand(n)], dim=1)


def _right_edge(n: int) -> Tensor:
    """x=1, y ~ U(0,1), t ~ U(0,1)."""
    return torch.stack([torch.ones(n), torch.rand(n), torch.rand(n)], dim=1)


def _bottom_edge(n: int) -> Tensor:
    """x ~ U(0,1), y=0, t ~ U(0,1)."""
    return torch.stack([torch.rand(n), torch.zeros(n), torch.rand(n)], dim=1)


def _top_edge(n: int) -> Tensor:
    """x ~ U(0,1), y=1, t ~ U(0,1)."""
    return torch.stack([torch.rand(n), torch.ones(n), torch.rand(n)], dim=1)


def _initial_condition(n: int) -> Tensor:
    """x ~ U(0,1), y ~ U(0,1), t=0."""
    return torch.stack([torch.rand(n), torch.rand(n), torch.zeros(n)], dim=1)


def _zero(x: Tensor) -> Tensor:
    """Homogeneous value: 0."""
    return torch.zeros(x.shape[0], 1)


def _ic_u(x: Tensor) -> Tensor:
    """IC for u: 1 everywhere, 0.5 in center square [0.4, 0.6]^2."""
    vals = torch.ones(x.shape[0], 1)
    center = (x[:, 0] >= 0.4) & (x[:, 0] <= 0.6) & (x[:, 1] >= 0.4) & (x[:, 1] <= 0.6)
    vals[center] = 0.5
    return vals


def _ic_v(x: Tensor) -> Tensor:
    """IC for v: 0 everywhere, 0.25 in center square [0.4, 0.6]^2."""
    vals = torch.zeros(x.shape[0], 1)
    center = (x[:, 0] >= 0.4) & (x[:, 0] <= 0.6) & (x[:, 1] >= 0.4) & (x[:, 1] <= 0.6)
    vals[center] = 0.25
    return vals


# ============================================================================
# Predict Data Function
# ============================================================================


def predict_data(x_data: Tensor, fields: FieldsRegistry, _params: ParamsRegistry) -> Tensor:
    """Predict u and v at data points. Returns (N, 2, 1)."""
    u_pred = fields[U_KEY](x_data)
    v_pred = fields[V_KEY](x_data)
    return torch.stack([u_pred, v_pred], dim=1)


# ============================================================================
# Data and Problem Factories
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
        # Neumann zero-flux BCs: 4 edges x 2 fields
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
        # ICs at t=0
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


# ============================================================================
# Plotting and Saving
# ============================================================================


def plot_and_save(
    predictions: Predictions,
    results_dir: Path,
    experiment_name: str,
) -> None:
    batch, preds, _trues = predictions
    xyt_data, uv_data = batch

    if xyt_data.ndim == 1:
        xyt_data = xyt_data.reshape(-1, 3)

    x_np = xyt_data[:, 0].numpy()
    y_np = xyt_data[:, 1].numpy()
    t_np = xyt_data[:, 2].numpy()
    u_pred = preds[U_KEY].numpy()
    v_pred = preds[V_KEY].numpy()
    u_true = uv_data[:, 0].squeeze(-1).numpy()
    v_true = uv_data[:, 1].squeeze(-1).numpy()

    # Recover parameters
    params_str = ""
    du_rec = preds.get(DU_KEY)
    dv_rec = preds.get(DV_KEY)
    f_rec = preds.get(F_KEY)
    k_rec = preds.get(K_KEY)
    if du_rec is not None:
        params_str = (
            f" | D_u={du_rec.mean().item():.4e} (true={TRUE_DU:.4e})"
            f", D_v={dv_rec.mean().item():.4e} (true={TRUE_DV:.4e})"
            f"\nF={f_rec.mean().item():.4e} (true={TRUE_F:.4e})"
            f", k={k_rec.mean().item():.4e} (true={TRUE_K:.4e})"
        )

    # Plot final time snapshot
    t_max = t_np.max()
    mask = np.isclose(t_np, t_max)
    x_final = x_np[mask]
    y_final = y_np[mask]
    u_p_final = u_pred[mask]
    u_t_final = u_true[mask]
    v_p_final = v_pred[mask]
    v_t_final = v_true[mask]

    n_side = int(np.sqrt(len(x_final)))
    X = x_final.reshape(n_side, n_side)
    Y = y_final.reshape(n_side, n_side)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"Gray-Scott 2D (t={t_max:.2f}){params_str}", fontsize=13)

    # Row 1: u
    u_grid = u_p_final.reshape(n_side, n_side)
    im00 = axes[0, 0].pcolormesh(X, Y, u_grid, shading="auto", cmap="viridis")
    axes[0, 0].set_title("Predicted $u$")
    axes[0, 0].set_xlabel("$x$")
    axes[0, 0].set_ylabel("$y$")
    axes[0, 0].set_aspect("equal")
    fig.colorbar(im00, ax=axes[0, 0])

    u_true_grid = u_t_final.reshape(n_side, n_side)
    im01 = axes[0, 1].pcolormesh(X, Y, u_true_grid, shading="auto", cmap="viridis")
    axes[0, 1].set_title("True $u$")
    axes[0, 1].set_xlabel("$x$")
    axes[0, 1].set_ylabel("$y$")
    axes[0, 1].set_aspect("equal")
    fig.colorbar(im01, ax=axes[0, 1])

    u_err = np.abs(u_p_final - u_t_final).reshape(n_side, n_side)
    im02 = axes[0, 2].pcolormesh(X, Y, u_err, shading="auto", cmap="hot")
    axes[0, 2].set_title("$|u_{pred} - u_{true}|$")
    axes[0, 2].set_xlabel("$x$")
    axes[0, 2].set_ylabel("$y$")
    axes[0, 2].set_aspect("equal")
    fig.colorbar(im02, ax=axes[0, 2])

    # Row 2: v
    v_grid = v_p_final.reshape(n_side, n_side)
    im10 = axes[1, 0].pcolormesh(X, Y, v_grid, shading="auto", cmap="viridis")
    axes[1, 0].set_title("Predicted $v$")
    axes[1, 0].set_xlabel("$x$")
    axes[1, 0].set_ylabel("$y$")
    axes[1, 0].set_aspect("equal")
    fig.colorbar(im10, ax=axes[1, 0])

    v_true_grid = v_t_final.reshape(n_side, n_side)
    im11 = axes[1, 1].pcolormesh(X, Y, v_true_grid, shading="auto", cmap="viridis")
    axes[1, 1].set_title("True $v$")
    axes[1, 1].set_xlabel("$x$")
    axes[1, 1].set_ylabel("$y$")
    axes[1, 1].set_aspect("equal")
    fig.colorbar(im11, ax=axes[1, 1])

    v_err = np.abs(v_p_final - v_t_final).reshape(n_side, n_side)
    im12 = axes[1, 2].pcolormesh(X, Y, v_err, shading="auto", cmap="hot")
    axes[1, 2].set_title("$|v_{pred} - v_{true}|$")
    axes[1, 2].set_xlabel("$x$")
    axes[1, 2].set_ylabel("$y$")
    axes[1, 2].set_aspect("equal")
    fig.colorbar(im12, ax=axes[1, 2])

    plt.tight_layout()
    fig.savefig(results_dir / f"{experiment_name}.png", dpi=300)
    plt.close(fig)

    df = pd.DataFrame(
        {
            "x": x_np,
            "y": y_np,
            "t": t_np,
            "u_pred": u_pred,
            "u_true": u_true,
            "v_pred": v_pred,
            "v_true": v_true,
        }
    )
    df.to_csv(results_dir / f"{experiment_name}.csv", index=False, float_format="%.6e")
