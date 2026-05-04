"""Poisson 2D — PDE problem definition."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    Predictions,
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
    xy_data, u_data = batch

    # xy_data may be (N,) after squeeze — reshape to (N, 2) if needed
    if xy_data.ndim == 1:
        xy_data = xy_data.reshape(-1, 2)

    x_np = xy_data[:, 0].numpy()
    y_np = xy_data[:, 1].numpy()
    u_pred = preds[U_KEY].numpy()
    u_true = u_data.reshape(-1).numpy()

    n_side = math.isqrt(x_np.shape[0])
    X = x_np.reshape(n_side, n_side)
    Y = y_np.reshape(n_side, n_side)
    U_pred = u_pred.reshape(n_side, n_side)
    U_true = u_true.reshape(n_side, n_side)
    error = np.abs(U_pred - U_true)

    k_recovered = preds.get(K_KEY)
    k_str = ""
    if k_recovered is not None:
        k_val = k_recovered.mean().item()
        k_str = (
            rf" | $k_{{\mathrm{{pred}}}} = {k_val:.4f}$,"
            rf" $k_{{\mathrm{{true}}}} = {TRUE_K}$"
        )

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Poisson Equation 2D{k_str}", fontsize=14)

    im0 = axes[0].pcolormesh(X, Y, U_pred, shading="auto", cmap="viridis")
    axes[0].set_title(r"Predicted $u(x,y)$")
    axes[0].set_xlabel(r"$x$")
    axes[0].set_ylabel(r"$y$")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(X, Y, U_true, shading="auto", cmap="viridis")
    axes[1].set_title(r"True $u(x,y)$")
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$y$")
    axes[1].set_aspect("equal")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(X, Y, error, shading="auto", cmap="hot")
    axes[2].set_title(r"Pointwise Error $|u_{\mathrm{pred}} - u_{\mathrm{true}}|$")
    axes[2].set_xlabel(r"$x$")
    axes[2].set_ylabel(r"$y$")
    axes[2].set_aspect("equal")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    fig.savefig(results_dir / "plot.png", dpi=300)
    plt.close(fig)

    df = pd.DataFrame(
        {
            "x": x_np,
            "y": y_np,
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
    xy_data, _u_data = batch

    if xy_data.ndim == 1:
        xy_data = xy_data.reshape(-1, 2)

    x_np = xy_data[:, 0].numpy()
    y_np = xy_data[:, 1].numpy()
    u_pred = preds[U_KEY].numpy()

    n_side = math.isqrt(x_np.shape[0])
    X = x_np.reshape(n_side, n_side)
    Y = y_np.reshape(n_side, n_side)
    U_pred = u_pred.reshape(n_side, n_side)

    k_recovered = preds.get(K_KEY)
    k_str = ""
    if k_recovered is not None:
        k_val = k_recovered.mean().item()
        k_str = rf" | $k_{{\mathrm{{pred}}}} = {k_val:.4f}$"

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle(f"Poisson Equation 2D{k_str}", fontsize=14)

    im0 = ax.pcolormesh(X, Y, U_pred, shading="auto", cmap="viridis")
    ax.set_title(r"Predicted $u(x,y)$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_aspect("equal")
    fig.colorbar(im0, ax=ax)

    plt.tight_layout()
    fig.savefig(results_dir / "plot.png", dpi=300)
    plt.close(fig)

    df = pd.DataFrame({"x": x_np, "y": y_np, "u_pred": u_pred})
    df.to_csv(results_dir / "data.csv", index=False, float_format="%.6e")


# --- END VARIANT ---
