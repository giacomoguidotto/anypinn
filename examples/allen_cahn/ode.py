"""Allen-Cahn Equation — stiff reaction-diffusion forward problem definition.

Solves: du/dt = eps * d2u/dx2 + u - u^3 on [-1,1] x [0,1]
BCs: periodic u(-1,t) = u(1,t), du/dx(-1,t) = du/dx(1,t)
IC: u(x,0) = -tanh(x / sqrt(2*eps))

Forward problem: no inverse parameters, no DataConstraint.
Demonstrates PeriodicBCConstraint, AdaptiveSampler, Huber criterion, and RFF.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor

from anypinn.catalog.allen_cahn import TRUE_EPSILON, U_KEY, AllenCahnDataModule
from anypinn.core import (
    Field,
    FieldsRegistry,
    MLPConfig,
    ParamsRegistry,
    PINNHyperparameters,
    Predictions,
    Problem,
    RandomFourierFeatures,
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


def allen_cahn_residual(x: Tensor, fields: FieldsRegistry, _params: ParamsRegistry) -> Tensor:
    """PDE residual: du/dt - eps*d2u/dx2 - u + u^3 = 0."""
    u = fields[U_KEY](x)
    du_dt = partial(u, x, dim=1, order=1)
    d2u_dx2 = partial(u, x, dim=0, order=2)
    return du_dt - EPSILON * d2u_dx2 - u + u**3


# ============================================================================
# Boundary / IC Samplers
# ============================================================================


def _periodic_left(n: int) -> Tensor:
    """x=-1, t ~ U(0,1). Paired with _periodic_right via shared RNG state."""
    t = torch.rand(n)
    return torch.stack([torch.full((n,), -1.0), t], dim=1)


def _periodic_right(n: int) -> Tensor:
    """x=+1, t ~ U(0,1). Paired with _periodic_left via shared RNG state."""
    t = torch.rand(n)
    return torch.stack([torch.ones(n), t], dim=1)


def _initial_condition(n: int) -> Tensor:
    """x ~ U(-1,1), t=0."""
    return torch.stack([2 * torch.rand(n) - 1, torch.zeros(n)], dim=1)


def _dummy(x: Tensor) -> Tensor:
    """Dummy value function (unused by PeriodicBCConstraint)."""
    return torch.zeros(x.shape[0], 1)


def _ic_value(x: Tensor) -> Tensor:
    """IC: u(x,0) = -tanh(x / sqrt(2*eps))."""
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
            res = allen_cahn_residual(x, self.fields, self.params)
        return res.detach().cpu().abs().mean(dim=-1)


# ============================================================================
# Data and Problem Factories
# ============================================================================


def create_data_module(
    hp: PINNHyperparameters,
    fields: FieldsRegistry,
    params: ParamsRegistry,
) -> AllenCahnDataModule:
    scorer = AllenCahnResidualScorer(fields, params)
    return AllenCahnDataModule(
        hp=hp,
        true_epsilon=TRUE_EPSILON,
        grid_size=GRID_SIZE,
        residual_scorer=scorer,
    )


def create_problem(hp: PINNHyperparameters) -> Problem:
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
        residual_fn=allen_cahn_residual,
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

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Allen-Cahn Equation", fontsize=14)

    im0 = axes[0].pcolormesh(X, T, U_pred, shading="auto", cmap="viridis")
    axes[0].set_title("Predicted $u(x,t)$")
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$t$")
    axes[0].set_aspect("equal")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(X, T, U_true, shading="auto", cmap="viridis")
    axes[1].set_title("Reference $u(x,t)$")
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
