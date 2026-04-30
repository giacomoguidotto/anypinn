"""Wave Equation 1D — inverse PDE problem definition.

Solves: d2u/dt2 = c^2 * d2u/dx2 on [0,1] x [0,1]
BCs: u(0,t) = u(1,t) = 0
ICs: u(x,0) = sin(pi*x), du/dt(x,0) = 0
Analytic solution: u(x,t) = sin(pi x) cos(c pi t)

Inverse problem: recover wave speed c from sparse interior measurements.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from anypinn.catalog.wave_1d import C_KEY, TRUE_C, U_KEY, Wave1DDataModule
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
# Constants
# ============================================================================

GRID_SIZE = 50


# ============================================================================
# PDE Definition
# ============================================================================


def wave_residual(x: Tensor, fields: FieldsRegistry, params: ParamsRegistry) -> Tensor:
    """PDE residual: d2u/dt2 - c^2 * d2u/dx2 = 0."""
    u = fields[U_KEY](x)
    c = params[C_KEY](x)
    d2u_dt2 = partial(u, x, dim=1, order=2)
    d2u_dx2 = partial(u, x, dim=0, order=2)
    return d2u_dt2 - c * c * d2u_dx2


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


validation: ValidationRegistry = {C_KEY: lambda x: torch.full_like(x, TRUE_C)}


def create_data_module(hp: PINNHyperparameters) -> Wave1DDataModule:
    return Wave1DDataModule(
        hp=hp,
        true_c=TRUE_C,
        grid_size=GRID_SIZE,
        validation=validation,
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
    param_c = Parameter(
        config=ScalarConfig(init_value=hp.params_config.init_value),
    )

    fields = FieldsRegistry({U_KEY: field_u})
    params = ParamsRegistry({C_KEY: param_c})

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
        NeumannBCConstraint(
            BoundaryCondition(sampler=_initial_condition, value=_zero, n_pts=100),
            field_u,
            normal_dim=1,
            log_key="loss/ic_velocity",
            weight=10.0,
        ),
    ]

    pde = PDEResidualConstraint(
        fields=fields,
        params=params,
        residual_fn=wave_residual,
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

    # Recover c from predictions metadata
    c_recovered = preds.get(C_KEY)
    c_str = ""
    if c_recovered is not None:
        c_val = c_recovered.mean().item()
        c_str = f" | Recovered c={c_val:.4f} (true={TRUE_C})"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Wave Equation 1D{c_str}", fontsize=14)

    im0 = axes[0].pcolormesh(X, T, U_pred, shading="auto", cmap="viridis")
    axes[0].set_title("Predicted $u(x,t)$")
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$t$")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(X, T, U_true, shading="auto", cmap="viridis")
    axes[1].set_title("Analytic $u(x,t)$")
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$t$")
    axes[1].set_aspect("equal")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(X, T, error, shading="auto", cmap="hot")
    axes[2].set_title("Pointwise Error $|u_{pred} - u_{true}|$")
    axes[2].set_xlabel("$x$")
    axes[2].set_ylabel("$t$")
    axes[2].set_aspect("equal")
    fig.colorbar(im2, ax=axes[2])

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
    df.to_csv(results_dir / f"{experiment_name}.csv", index=False, float_format="%.6e")
