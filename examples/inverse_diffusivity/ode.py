"""Inverse Diffusivity — inverse PDE problem definition.

Solves: du/dt = div(D(x) grad(u)) on [0,1] x [0,1]
Expanded in 1D: du/dt = dD/dx * du/dx + D * d2u/dx2
BCs: u(0,t) = u(1,t) = 0
IC: u(x,0) = sin(pi*x)

Inverse problem: recover the spatially varying diffusivity D(x)
from sparse interior measurements. D(x) is modelled as a neural
network Field (not a scalar Parameter).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from anypinn.catalog.inverse_diffusivity import (
    D_KEY,
    U_KEY,
    InverseDiffusivityDataModule,
    true_d_fn,
)
from anypinn.core import (
    Field,
    FieldsRegistry,
    FourierEncoding,
    MLPConfig,
    ParamsRegistry,
    PINNHyperparameters,
    Predictions,
    Problem,
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


def diffusivity_residual(x: Tensor, fields: FieldsRegistry, params: ParamsRegistry) -> Tensor:
    """PDE residual: du/dt - (dD/dx * du/dx + D * d2u/dx2) = 0."""
    u = fields[U_KEY](x)
    D = fields[D_KEY](x)
    du_dt = partial(u, x, dim=1, order=1)
    du_dx = partial(u, x, dim=0, order=1)
    d2u_dx2 = partial(u, x, dim=0, order=2)
    dD_dx = partial(D, x, dim=0, order=1)
    return du_dt - (dD_dx * du_dx + D * d2u_dx2)


# ============================================================================
# Boundary / IC Samplers
# ============================================================================


def _left_boundary(n: int) -> Tensor:
    """x=0, t ~ U(0,1)."""
    return torch.stack([torch.zeros(n), torch.rand(n)], dim=1)


def _right_boundary(n: int) -> Tensor:
    """x=1, t ~ U(0,1)."""
    return torch.stack([torch.ones(n), torch.rand(n)], dim=1)


def _initial_condition(n: int) -> Tensor:
    """x ~ U(0,1), t=0."""
    return torch.stack([torch.rand(n), torch.zeros(n)], dim=1)


def _zero(x: Tensor) -> Tensor:
    """Homogeneous Dirichlet: u = 0."""
    return torch.zeros(x.shape[0], 1)


def _ic_value(x: Tensor) -> Tensor:
    """IC: u(x,0) = sin(pi*x)."""
    return torch.sin(math.pi * x[:, 0:1])


# ============================================================================
# Predict Data Function
# ============================================================================


def predict_data(x_data: Tensor, fields: FieldsRegistry, _params: ParamsRegistry) -> Tensor:
    """Predict u at data points. Returns (N, 1, 1)."""
    return fields[U_KEY](x_data).unsqueeze(1)


# ============================================================================
# Data and Problem Factories
# ============================================================================


def create_data_module(hp: PINNHyperparameters) -> InverseDiffusivityDataModule:
    return InverseDiffusivityDataModule(
        hp=hp,
        grid_size=GRID_SIZE,
    )


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
    field_d = Field(
        config=MLPConfig(
            in_dim=encode.out_dim(2),
            out_dim=1,
            hidden_layers=hp.fields_config.hidden_layers,
            activation=hp.fields_config.activation,
            output_activation="softplus",
            encode=encode,
        )
    )

    fields = FieldsRegistry({U_KEY: field_u, D_KEY: field_d})
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
        residual_fn=diffusivity_residual,
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


# ============================================================================
# Plotting and Saving
# ============================================================================


def plot_and_save(
    predictions: Predictions,
    results_dir: Path,
    experiment_name: str,
) -> None:
    batch, preds, _trues = predictions
    xt_data, u_data = batch

    # xt_data may be (N,) after squeeze — reshape to (N, 2) if needed
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

    # Recover D(x) from predictions
    d_pred = preds.get(D_KEY)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Inverse Diffusivity: $\\partial u/\\partial t = \\nabla\\cdot(D(x)\\nabla u)$",
        fontsize=14,
    )

    im0 = axes[0].pcolormesh(X, T, U_pred, shading="auto", cmap="viridis")
    axes[0].set_title("Predicted $u(x,t)$")
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$t$")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(X, T, error, shading="auto", cmap="hot")
    axes[1].set_title("Pointwise Error $|u_{pred} - u_{true}|$")
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$t$")
    axes[1].set_aspect("equal")
    fig.colorbar(im1, ax=axes[1])

    # D(x) line plot: recovered vs true
    x_line = np.linspace(0, 1, 200)
    d_true = 0.1 + 0.05 * np.sin(2 * np.pi * x_line)
    axes[2].plot(x_line, d_true, "k--", label="True $D(x)$", linewidth=2)

    if d_pred is not None:
        # Average D predictions over time (D should be independent of t)
        d_pred_np = d_pred.numpy()
        d_grid = d_pred_np.reshape(n_side, n_side)
        d_mean = d_grid.mean(axis=1)  # average over t
        x_grid_unique = X[:, 0]
        axes[2].plot(x_grid_unique, d_mean, "r-", label="Recovered $D(x)$", linewidth=2)

    axes[2].set_title("Diffusivity $D(x)$")
    axes[2].set_xlabel("$x$")
    axes[2].set_ylabel("$D$")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(results_dir / f"{experiment_name}.png", dpi=300)
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
    if d_pred is not None:
        df["D_pred"] = d_pred.numpy()
        df["D_true"] = true_d_fn(xt_data[:, 0:1]).squeeze().numpy()
    df.to_csv(results_dir / f"{experiment_name}.csv", index=False, float_format="%.6e")
