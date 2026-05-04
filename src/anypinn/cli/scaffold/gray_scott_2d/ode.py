"""Gray-Scott 2D Reaction-Diffusion — PDE problem definition."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    Predictions,
    Problem,
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
    NeumannBCConstraint,
    PDEResidualConstraint,
)

# ============================================================================
# PDE Definition
# ============================================================================


# --- VARIANT: direction/forward ---
def gs_residual_u_forward(x: Tensor, fields: FieldsRegistry, _params: ParamsRegistry) -> Tensor:
    """PDE residual for u: du/dt_norm - T * (D_u lap_u - uv^2 + F(1-u)) = 0 (known)."""
    u = fields[U_KEY](x)
    v = fields[V_KEY](x)
    du_dt = partial(u, x, dim=2, order=1)
    lap_u = partial(u, x, dim=0, order=2) + partial(u, x, dim=1, order=2)
    return du_dt - (TRUE_DU * lap_u - u * v**2 + TRUE_F * (1 - u)) * T_TOTAL


def gs_residual_v_forward(x: Tensor, fields: FieldsRegistry, _params: ParamsRegistry) -> Tensor:
    """PDE residual for v: dv/dt_norm - T * (D_v lap_v + uv^2 - (F+k)v) = 0 (known)."""
    u = fields[U_KEY](x)
    v = fields[V_KEY](x)
    dv_dt = partial(v, x, dim=2, order=1)
    lap_v = partial(v, x, dim=0, order=2) + partial(v, x, dim=1, order=2)
    return dv_dt - (TRUE_DV * lap_v + u * v**2 - (TRUE_F + TRUE_K) * v) * T_TOTAL


# --- VARIANT: direction/inverse ---
def gs_residual_u_inverse(x: Tensor, fields: FieldsRegistry, params: ParamsRegistry) -> Tensor:
    """PDE residual for u: du/dt_norm - T * (D_u lap_u - uv^2 + F(1-u)) = 0 (learned)."""
    u = fields[U_KEY](x)
    v = fields[V_KEY](x)
    du = torch.nn.functional.softplus(params[DU_KEY](x))
    f = torch.nn.functional.softplus(params[F_KEY](x))
    du_dt = partial(u, x, dim=2, order=1)
    lap_u = partial(u, x, dim=0, order=2) + partial(u, x, dim=1, order=2)
    return du_dt - (du * lap_u - u * v**2 + f * (1 - u)) * T_TOTAL


def gs_residual_v_inverse(x: Tensor, fields: FieldsRegistry, params: ParamsRegistry) -> Tensor:
    """PDE residual for v: dv/dt_norm - T * (D_v lap_v + uv^2 - (F+k)v) = 0 (learned)."""
    u = fields[U_KEY](x)
    v = fields[V_KEY](x)
    dv = torch.nn.functional.softplus(params[DV_KEY](x))
    f = torch.nn.functional.softplus(params[F_KEY](x))
    k = torch.nn.functional.softplus(params[K_KEY](x))
    dv_dt = partial(v, x, dim=2, order=1)
    lap_v = partial(v, x, dim=0, order=2) + partial(v, x, dim=1, order=2)
    return dv_dt - (dv * lap_v + u * v**2 - (f + k) * v) * T_TOTAL


# --- END VARIANT ---

# ============================================================================
# Boundary / IC Samplers
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
    return torch.zeros(x.shape[0], 1)


def _ic_u(x: Tensor) -> Tensor:
    vals = torch.ones(x.shape[0], 1)
    center = (x[:, 0] >= 0.4) & (x[:, 0] <= 0.6) & (x[:, 1] >= 0.4) & (x[:, 1] <= 0.6)
    vals[center] = 0.5
    return vals


def _ic_v(x: Tensor) -> Tensor:
    vals = torch.zeros(x.shape[0], 1)
    center = (x[:, 0] >= 0.4) & (x[:, 0] <= 0.6) & (x[:, 1] >= 0.4) & (x[:, 1] <= 0.6)
    vals[center] = 0.25
    return vals


# --- VARIANT: direction/inverse ---
# ============================================================================
# Predict Data Function
# ============================================================================


def predict_data(x_data: Tensor, fields: FieldsRegistry, _params: ParamsRegistry) -> Tensor:
    u_pred = fields[U_KEY](x_data)
    v_pred = fields[V_KEY](x_data)
    return torch.stack([u_pred, v_pred], dim=1)


# --- END VARIANT ---

# ============================================================================
# Data Module Factory
# ============================================================================

# --- VARIANT: direction/inverse ---
_validation: ValidationRegistry = {
    DU_KEY: lambda x: torch.full_like(x, TRUE_DU),
    DV_KEY: lambda x: torch.full_like(x, TRUE_DV),
    F_KEY: lambda x: torch.full_like(x, TRUE_F),
    K_KEY: lambda x: torch.full_like(x, TRUE_K),
}
# --- VARIANT: direction/forward ---
_validation = None
# --- END VARIANT ---


# --- VARIANT: source/synthetic ---
def create_data_module_synthetic(hp: PINNHyperparameters) -> GrayScott2DDataModule:
    return GrayScott2DDataModule(
        hp=hp,
        true_du=TRUE_DU,
        true_dv=TRUE_DV,
        true_f=TRUE_F,
        true_k=TRUE_K,
        validation=_validation,
    )


# --- VARIANT: source/csv ---
def create_data_module_csv(hp: PINNHyperparameters) -> GrayScott2DDataModule:
    return GrayScott2DDataModule(
        hp=hp,
        true_du=TRUE_DU,
        true_dv=TRUE_DV,
        true_f=TRUE_F,
        true_k=TRUE_K,
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

    fields = FieldsRegistry({U_KEY: field_u, V_KEY: field_v})
    params = ParamsRegistry({})

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
        residual_fn=gs_residual_u_forward,
        log_key="loss/pde_u",
        weight=1e-4,
    )
    pde_v = PDEResidualConstraint(
        fields=fields,
        params=params,
        residual_fn=gs_residual_v_forward,
        log_key="loss/pde_v",
        weight=1e-4,
    )

    return Problem(
        constraints=[pde_u, pde_v, *bcs],
        criterion=build_criterion(hp.criterion),
        fields=fields,
        params=params,
    )


# --- VARIANT: direction/inverse ---
def create_problem_inverse(hp: PINNHyperparameters) -> Problem:
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
        residual_fn=gs_residual_u_inverse,
        log_key="loss/pde_u",
        weight=1e-4,
    )
    pde_v = PDEResidualConstraint(
        fields=fields,
        params=params,
        residual_fn=gs_residual_v_inverse,
        log_key="loss/pde_v",
        weight=1e-4,
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


# --- END VARIANT ---


# ============================================================================
# Plotting and Saving
# ============================================================================


# --- VARIANT: source/synthetic ---
def plot_and_save_synthetic(
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

    # Recover parameters (apply softplus to match PDE usage)
    params_str = ""
    du_rec = preds.get(DU_KEY)
    dv_rec = preds.get(DV_KEY)
    f_rec = preds.get(F_KEY)
    k_rec = preds.get(K_KEY)
    if du_rec is not None:
        du_val = torch.nn.functional.softplus(torch.tensor(du_rec.mean())).item()
        dv_val = torch.nn.functional.softplus(torch.tensor(dv_rec.mean())).item()
        f_val = torch.nn.functional.softplus(torch.tensor(f_rec.mean())).item()
        k_val = torch.nn.functional.softplus(torch.tensor(k_rec.mean())).item()
        params_str = (
            rf" | $D_{{u,\mathrm{{pred}}}} = {du_val:.4e}$"
            rf", $D_{{v,\mathrm{{pred}}}} = {dv_val:.4e}$"
            rf", $F_{{\mathrm{{pred}}}} = {f_val:.4e}$"
            rf", $k_{{\mathrm{{pred}}}} = {k_val:.4e}$"
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

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"Gray--Scott 2D ($t = {t_max:.2f}$){params_str}", fontsize=13)

    u_grid = u_p_final.reshape(n_side, n_side)
    im00 = axes[0, 0].pcolormesh(X, Y, u_grid, shading="auto", cmap="viridis")
    axes[0, 0].set_title(r"Predicted $u$")
    axes[0, 0].set_xlabel(r"$x$")
    axes[0, 0].set_ylabel(r"$y$")
    axes[0, 0].set_aspect("equal")
    fig.colorbar(im00, ax=axes[0, 0])

    u_true_grid = u_t_final.reshape(n_side, n_side)
    im01 = axes[0, 1].pcolormesh(X, Y, u_true_grid, shading="auto", cmap="viridis")
    axes[0, 1].set_title(r"True $u$")
    axes[0, 1].set_xlabel(r"$x$")
    axes[0, 1].set_ylabel(r"$y$")
    axes[0, 1].set_aspect("equal")
    fig.colorbar(im01, ax=axes[0, 1])

    u_err = np.abs(u_p_final - u_t_final).reshape(n_side, n_side)
    im02 = axes[0, 2].pcolormesh(X, Y, u_err, shading="auto", cmap="hot")
    axes[0, 2].set_title(r"$|u_{\mathrm{pred}} - u_{\mathrm{true}}|$")
    axes[0, 2].set_xlabel(r"$x$")
    axes[0, 2].set_ylabel(r"$y$")
    axes[0, 2].set_aspect("equal")
    fig.colorbar(im02, ax=axes[0, 2])

    v_grid = v_p_final.reshape(n_side, n_side)
    im10 = axes[1, 0].pcolormesh(X, Y, v_grid, shading="auto", cmap="viridis")
    axes[1, 0].set_title(r"Predicted $v$")
    axes[1, 0].set_xlabel(r"$x$")
    axes[1, 0].set_ylabel(r"$y$")
    axes[1, 0].set_aspect("equal")
    fig.colorbar(im10, ax=axes[1, 0])

    v_true_grid = v_t_final.reshape(n_side, n_side)
    im11 = axes[1, 1].pcolormesh(X, Y, v_true_grid, shading="auto", cmap="viridis")
    axes[1, 1].set_title(r"True $v$")
    axes[1, 1].set_xlabel(r"$x$")
    axes[1, 1].set_ylabel(r"$y$")
    axes[1, 1].set_aspect("equal")
    fig.colorbar(im11, ax=axes[1, 1])

    v_err = np.abs(v_p_final - v_t_final).reshape(n_side, n_side)
    im12 = axes[1, 2].pcolormesh(X, Y, v_err, shading="auto", cmap="hot")
    axes[1, 2].set_title(r"$|v_{\mathrm{pred}} - v_{\mathrm{true}}|$")
    axes[1, 2].set_xlabel(r"$x$")
    axes[1, 2].set_ylabel(r"$y$")
    axes[1, 2].set_aspect("equal")
    fig.colorbar(im12, ax=axes[1, 2])

    plt.tight_layout()
    fig.savefig(results_dir / "plot.png", dpi=300)
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
    df.to_csv(results_dir / "data.csv", index=False, float_format="%.6e")


# --- VARIANT: source/csv ---
def plot_and_save_csv(
    predictions: Predictions,
    results_dir: Path,
    experiment_name: str,
) -> None:
    batch, preds, _trues = predictions
    xyt_data, _uv_data = batch

    if xyt_data.ndim == 1:
        xyt_data = xyt_data.reshape(-1, 3)

    x_np = xyt_data[:, 0].numpy()
    y_np = xyt_data[:, 1].numpy()
    t_np = xyt_data[:, 2].numpy()
    u_pred = preds[U_KEY].numpy()
    v_pred = preds[V_KEY].numpy()

    # Recover parameters (apply softplus to match PDE usage)
    params_str = ""
    du_rec = preds.get(DU_KEY)
    dv_rec = preds.get(DV_KEY)
    f_rec = preds.get(F_KEY)
    k_rec = preds.get(K_KEY)
    if du_rec is not None:
        du_val = torch.nn.functional.softplus(torch.tensor(du_rec.mean())).item()
        dv_val = torch.nn.functional.softplus(torch.tensor(dv_rec.mean())).item()
        f_val = torch.nn.functional.softplus(torch.tensor(f_rec.mean())).item()
        k_val = torch.nn.functional.softplus(torch.tensor(k_rec.mean())).item()
        params_str = (
            rf" | $D_{{u,\mathrm{{pred}}}} = {du_val:.4e}$"
            rf", $D_{{v,\mathrm{{pred}}}} = {dv_val:.4e}$"
            rf", $F_{{\mathrm{{pred}}}} = {f_val:.4e}$"
            rf", $k_{{\mathrm{{pred}}}} = {k_val:.4e}$"
        )

    # Plot final time snapshot
    t_max = t_np.max()
    mask = np.isclose(t_np, t_max)
    x_final = x_np[mask]
    y_final = y_np[mask]
    u_p_final = u_pred[mask]
    v_p_final = v_pred[mask]

    n_side = int(np.sqrt(len(x_final)))
    X = x_final.reshape(n_side, n_side)
    Y = y_final.reshape(n_side, n_side)

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(2, 1, figsize=(8, 11))
    fig.suptitle(f"Gray--Scott 2D ($t = {t_max:.2f}$){params_str}", fontsize=13)

    u_grid = u_p_final.reshape(n_side, n_side)
    im0 = axes[0].pcolormesh(X, Y, u_grid, shading="auto", cmap="viridis")
    axes[0].set_title(r"Predicted $u$")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$y$")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0])

    v_grid = v_p_final.reshape(n_side, n_side)
    im1 = axes[1].pcolormesh(X, Y, v_grid, shading="auto", cmap="viridis")
    axes[1].set_title(r"Predicted $v$")
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$y$")
    axes[1].set_aspect("equal")
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    fig.savefig(results_dir / "plot.png", dpi=300)
    plt.close(fig)

    df = pd.DataFrame({"x": x_np, "y": y_np, "t": t_np, "u_pred": u_pred, "v_pred": v_pred})
    df.to_csv(results_dir / "data.csv", index=False, float_format="%.6e")


# --- END VARIANT ---
