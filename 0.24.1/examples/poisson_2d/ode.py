"""Poisson 2D — forward PDE problem definition.

Solves: nabla^2 u = f(x,y) on [0,1]^2, u = 0 on boundary
Source: f(x,y) = -2 pi^2 sin(pi*x) sin(pi*y)
Analytic solution: u(x,y) = sin(pi*x) sin(pi*y)
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from anypinn.catalog.poisson_2d import U_KEY, Poisson2DDataModule
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


def poisson_residual(x: Tensor, fields: FieldsRegistry, params: ParamsRegistry) -> Tensor:
    """PDE residual: laplacian(u) - f = 0."""
    from anypinn.lib.diff import laplacian

    u = fields[U_KEY](x)
    return laplacian(u, x) - source_fn(x)


# ============================================================================
# Boundary Samplers
# ============================================================================


def _left_edge(n: int) -> Tensor:
    """x=0, y ~ U(0,1)."""
    return torch.stack([torch.zeros(n), torch.rand(n)], dim=1)


def _right_edge(n: int) -> Tensor:
    """x=1, y ~ U(0,1)."""
    return torch.stack([torch.ones(n), torch.rand(n)], dim=1)


def _bottom_edge(n: int) -> Tensor:
    """x ~ U(0,1), y=0."""
    return torch.stack([torch.rand(n), torch.zeros(n)], dim=1)


def _top_edge(n: int) -> Tensor:
    """x ~ U(0,1), y=1."""
    return torch.stack([torch.rand(n), torch.ones(n)], dim=1)


def _zero(x: Tensor) -> Tensor:
    """Homogeneous Dirichlet: u = 0."""
    return torch.zeros(x.shape[0], 1)


# ============================================================================
# Data and Problem Factories
# ============================================================================


def create_data_module(hp: PINNHyperparameters) -> Poisson2DDataModule:
    return Poisson2DDataModule(hp=hp, grid_size=GRID_SIZE)


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
        residual_fn=poisson_residual,
        log_key="loss/pde_residual",
        weight=1.0,
    )

    return Problem(
        constraints=[pde, *bcs],
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

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im0 = axes[0].pcolormesh(X, Y, U_pred, shading="auto", cmap="viridis")
    axes[0].set_title("Predicted $u(x,y)$")
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(X, Y, U_true, shading="auto", cmap="viridis")
    axes[1].set_title("Analytic $u(x,y)$")
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$y$")
    axes[1].set_aspect("equal")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(X, Y, error, shading="auto", cmap="hot")
    axes[2].set_title("Pointwise Error $|u_{pred} - u_{true}|$")
    axes[2].set_xlabel("$x$")
    axes[2].set_ylabel("$y$")
    axes[2].set_aspect("equal")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    fig.savefig(results_dir / f"{experiment_name}.png", dpi=300)
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
    df.to_csv(results_dir / f"{experiment_name}.csv", index=False, float_format="%.6e")
