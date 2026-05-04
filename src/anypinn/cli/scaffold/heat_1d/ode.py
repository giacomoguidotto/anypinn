"""Heat Equation 1D — PDE problem definition."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import Tensor

from anypinn.catalog.heat_1d import ALPHA_KEY, TRUE_ALPHA, U_KEY, Heat1DDataModule
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
from anypinn.problems import BoundaryCondition, DirichletBCConstraint, PDEResidualConstraint

# ============================================================================
# Constants
# ============================================================================

GRID_SIZE = 50

# ============================================================================
# PDE Definition
# ============================================================================


# --- VARIANT: direction/forward ---
def heat_residual_forward(x: Tensor, fields: FieldsRegistry, _params: ParamsRegistry) -> Tensor:
    """PDE residual: du/dt - alpha * d2u/dx2 = 0 (alpha known)."""
    u = fields[U_KEY](x)
    du_dt = partial(u, x, dim=1, order=1)
    d2u_dx2 = partial(u, x, dim=0, order=2)
    return du_dt - TRUE_ALPHA * d2u_dx2


# --- VARIANT: direction/inverse ---
def heat_residual_inverse(x: Tensor, fields: FieldsRegistry, params: ParamsRegistry) -> Tensor:
    """PDE residual: du/dt - alpha * d2u/dx2 = 0 (alpha learned)."""
    u = fields[U_KEY](x)
    alpha = params[ALPHA_KEY](x)
    du_dt = partial(u, x, dim=1, order=1)
    d2u_dx2 = partial(u, x, dim=0, order=2)
    return du_dt - alpha * d2u_dx2


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
_validation: ValidationRegistry = {ALPHA_KEY: lambda x: torch.full_like(x, TRUE_ALPHA)}
# --- VARIANT: direction/forward ---
_validation = None
# --- END VARIANT ---


# --- VARIANT: source/synthetic ---
def create_data_module_synthetic(hp: PINNHyperparameters) -> Heat1DDataModule:
    return Heat1DDataModule(
        hp=hp,
        true_alpha=TRUE_ALPHA,
        grid_size=GRID_SIZE,
        validation=_validation,
    )


# --- VARIANT: source/csv ---
def create_data_module_csv(hp: PINNHyperparameters) -> Heat1DDataModule:
    return Heat1DDataModule(
        hp=hp,
        true_alpha=TRUE_ALPHA,
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
        residual_fn=heat_residual_forward,
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
        residual_fn=heat_residual_inverse,
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
    xt_data, u_data = batch

    if xt_data.ndim == 1:
        xt_data = xt_data.reshape(-1, 2)

    x_np = xt_data[:, 0].numpy()
    t_np = xt_data[:, 1].numpy()
    u_pred = preds[U_KEY].numpy()
    u_true = u_data.reshape(-1).numpy()

    n_side = math.isqrt(x_np.shape[0])
    X = x_np.reshape(n_side, n_side)
    T = t_np.reshape(n_side, n_side)
    U_pred = u_pred.reshape(n_side, n_side)
    U_true = u_true.reshape(n_side, n_side)
    error = np.abs(U_pred - U_true)

    alpha_recovered = preds.get(ALPHA_KEY)
    alpha_str = ""
    if alpha_recovered is not None:
        alpha_val = alpha_recovered.mean().item()
        alpha_str = (
            rf" | $\alpha_{{\mathrm{{pred}}}} = {alpha_val:.4f}$,"
            rf" $\alpha_{{\mathrm{{true}}}} = {TRUE_ALPHA}$"
        )

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Heat Equation 1D{alpha_str}", fontsize=14)

    im0 = axes[0].pcolormesh(X, T, U_pred, shading="auto", cmap="viridis")
    axes[0].set_title(r"Predicted $u(x,t)$")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$t$")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(X, T, U_true, shading="auto", cmap="viridis")
    axes[1].set_title(r"True $u(x,t)$")
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$t$")
    axes[1].set_aspect("equal")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(X, T, error, shading="auto", cmap="hot")
    axes[2].set_title(r"Pointwise Error $|u_{\mathrm{pred}} - u_{\mathrm{true}}|$")
    axes[2].set_xlabel(r"$x$")
    axes[2].set_ylabel(r"$t$")
    axes[2].set_aspect("equal")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    fig.savefig(results_dir / "plot.png", dpi=300)
    plt.close(fig)

    df = pd.DataFrame(
        {
            "x": x_np,
            "t": t_np,
            "u_pred": u_pred,
            "u_true": u_true,
            "error": error.reshape(-1),
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
    xt_data, _u_data = batch

    if xt_data.ndim == 1:
        xt_data = xt_data.reshape(-1, 2)

    x_np = xt_data[:, 0].numpy()
    t_np = xt_data[:, 1].numpy()
    u_pred = preds[U_KEY].numpy()

    n_side = math.isqrt(x_np.shape[0])
    X = x_np.reshape(n_side, n_side)
    T = t_np.reshape(n_side, n_side)
    U_pred = u_pred.reshape(n_side, n_side)

    alpha_recovered = preds.get(ALPHA_KEY)
    alpha_str = ""
    if alpha_recovered is not None:
        alpha_val = alpha_recovered.mean().item()
        alpha_str = rf" | $\alpha_{{\mathrm{{pred}}}} = {alpha_val:.4f}$"

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle(f"Heat Equation 1D{alpha_str}", fontsize=14)

    im0 = ax.pcolormesh(X, T, U_pred, shading="auto", cmap="viridis")
    ax.set_title(r"Predicted $u(x,t)$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$t$")
    ax.set_aspect("equal")
    fig.colorbar(im0, ax=ax)

    plt.tight_layout()
    fig.savefig(results_dir / "plot.png", dpi=300)
    plt.close(fig)

    df = pd.DataFrame({"x": x_np, "t": t_np, "u_pred": u_pred})
    df.to_csv(results_dir / "data.csv", index=False, float_format="%.6e")


# --- END VARIANT ---
