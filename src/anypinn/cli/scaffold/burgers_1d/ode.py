"""Burgers Equation 1D — nonlinear PDE problem definition."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import Tensor

from anypinn.catalog.burgers_1d import NU_KEY, TRUE_NU, U_KEY, Burgers1DDataModule
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
    Predictions,
    Problem,
    RandomFourierFeatures,
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
def burgers_residual_forward(x: Tensor, fields: FieldsRegistry, _params: ParamsRegistry) -> Tensor:
    """PDE residual: du/dt + u*du/dx - nu*d2u/dx2 = 0 (nu known)."""
    u = fields[U_KEY](x)
    du_dt = partial(u, x, dim=1, order=1)
    du_dx = partial(u, x, dim=0, order=1)
    d2u_dx2 = partial(u, x, dim=0, order=2)
    return du_dt + u * du_dx - TRUE_NU * d2u_dx2


# --- VARIANT: direction/inverse ---
def burgers_residual_inverse(x: Tensor, fields: FieldsRegistry, params: ParamsRegistry) -> Tensor:
    """PDE residual: du/dt + u*du/dx - nu*d2u/dx2 = 0 (nu learned)."""
    u = fields[U_KEY](x)
    nu = params[NU_KEY](x)
    du_dt = partial(u, x, dim=1, order=1)
    du_dx = partial(u, x, dim=0, order=1)
    d2u_dx2 = partial(u, x, dim=0, order=2)
    return du_dt + u * du_dx - nu * d2u_dx2


# --- END VARIANT ---

# ============================================================================
# Boundary / IC Samplers
# ============================================================================


def _left_boundary(n: int) -> Tensor:
    return torch.stack([torch.full((n,), -1.0), torch.rand(n)], dim=1)


def _right_boundary(n: int) -> Tensor:
    return torch.stack([torch.ones(n), torch.rand(n)], dim=1)


def _initial_condition(n: int) -> Tensor:
    return torch.stack([2 * torch.rand(n) - 1, torch.zeros(n)], dim=1)


def _zero(x: Tensor) -> Tensor:
    return torch.zeros(x.shape[0], 1)


def _ic_value(x: Tensor) -> Tensor:
    return -torch.sin(math.pi * x[:, 0:1])


# --- VARIANT: direction/inverse ---
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
            res = burgers_residual_inverse(x, self.fields, self.params)
        # res may be (n, d) due to scalar param broadcasting; reduce to (n,)
        return res.detach().cpu().abs().mean(dim=-1)


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
_validation: ValidationRegistry = {NU_KEY: lambda x: torch.full_like(x, TRUE_NU)}
# --- VARIANT: direction/forward ---
_validation = None
# --- END VARIANT ---


def create_data_module(hp: PINNHyperparameters) -> Burgers1DDataModule:
    # To enable adaptive collocation, pass residual_scorer=BurgersResidualScorer(fields, params)
    return Burgers1DDataModule(
        hp=hp,
        true_nu=TRUE_NU,
        grid_size=GRID_SIZE,
        validation=_validation,
    )


# ============================================================================
# Problem Factory
# ============================================================================


# --- VARIANT: direction/forward ---
def create_problem_forward(hp: PINNHyperparameters) -> Problem:
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
        residual_fn=burgers_residual_forward,
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
        residual_fn=burgers_residual_inverse,
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

    nu_recovered = preds.get(NU_KEY)
    nu_str = ""
    if nu_recovered is not None:
        nu_val = nu_recovered.mean().item()
        nu_str = (
            rf" | $\nu_{{\mathrm{{pred}}}} = {nu_val:.6f}$,"
            rf" $\nu_{{\mathrm{{true}}}} = {TRUE_NU:.6f}$"
        )

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Burgers Equation 1D{nu_str}", fontsize=14)

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

    nu_recovered = preds.get(NU_KEY)
    nu_str = ""
    if nu_recovered is not None:
        nu_val = nu_recovered.mean().item()
        nu_str = rf" | $\nu_{{\mathrm{{pred}}}} = {nu_val:.6f}$"

    sns.set_theme(style="darkgrid")
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle(f"Burgers Equation 1D{nu_str}", fontsize=14)

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
