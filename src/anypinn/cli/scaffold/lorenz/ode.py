"""Lorenz system — mathematical definition."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import Tensor

from anypinn.core import (
    ArgsRegistry,
    Argument,
    # --- VARIANT: source/csv ---
    ColumnRef,  # noqa: F401
    # --- END VARIANT ---
    Field,
    FieldsRegistry,
    Parameter,
    ParamsRegistry,
    Predictions,
    ValidationRegistry,
)
from anypinn.lightning.callbacks import DataScaling
from anypinn.problems import ODEHyperparameters, ODEInverseProblem, ODEProperties

# ============================================================================
# Keys
# ============================================================================

X_KEY = "x"
Y_KEY = "y"
Z_KEY = "z"
SIGMA_KEY = "sigma"
RHO_KEY = "rho"
BETA_KEY = "beta"

# ============================================================================
# Constants
# ============================================================================

# True parameter values
TRUE_SIGMA = 10.0
TRUE_RHO = 28.0
TRUE_BETA = 8.0 / 3.0

# Initial conditions
X0 = -8.0
Y0 = 7.0
Z0 = 27.0

# Time domain
T_TOTAL = 3

# State scaling (Lorenz variables range ~±20-45)
SCALE = 20.0

# Noise level for synthetic data (additive Gaussian)
NOISE_STD = 0.5

# ============================================================================
# ODE Definition
# ============================================================================


def lorenz_scaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    """Scaled Lorenz ODE. States pre-divided by SCALE, time by T_TOTAL."""
    lx, ly, lz = y
    sigma = args[SIGMA_KEY]
    rho = args[RHO_KEY]
    beta = args[BETA_KEY]

    dx = sigma(x) * (ly - lx)
    dy = lx * (rho(x) / SCALE - lz) - ly
    dz = lx * ly * SCALE - beta(x) * lz

    dx = dx * T_TOTAL
    dy = dy * T_TOTAL
    dz = dz * T_TOTAL
    return torch.stack([dx, dy, dz])


# ============================================================================
# Validation
# ============================================================================

# --- VARIANT: source/synthetic ---
validation_synthetic: ValidationRegistry = {
    SIGMA_KEY: lambda x: torch.full_like(x, TRUE_SIGMA),
    RHO_KEY: lambda x: torch.full_like(x, TRUE_RHO),
    BETA_KEY: lambda x: torch.full_like(x, TRUE_BETA),
}
# --- VARIANT: source/csv ---
validation_csv: ValidationRegistry = {
    # TODO: map parameters to columns in your CSV
    # SIGMA_KEY: ColumnRef(column="your_column"),
    # RHO_KEY: ColumnRef(column="your_column"),
    # BETA_KEY: ColumnRef(column="your_column"),
}
# --- END VARIANT ---

# ============================================================================
# Data Module Factory
# ============================================================================


# --- VARIANT: source/synthetic ---
def create_data_module_synthetic(hp: ODEHyperparameters):
    from anypinn.catalog.lorenz import LorenzDataModule

    def lorenz_unscaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
        lx, ly, lz = y
        sigma = args[SIGMA_KEY]
        rho = args[RHO_KEY]
        beta = args[BETA_KEY]
        dx = sigma(x) * (ly - lx)
        dy = lx * (rho(x) - lz) - ly
        dz = lx * ly - beta(x) * lz
        return torch.stack([dx, dy, dz])

    gen_props = ODEProperties(
        ode=lorenz_unscaled,
        y0=torch.tensor([X0, Y0, Z0]),
        args={
            SIGMA_KEY: Argument(TRUE_SIGMA),
            RHO_KEY: Argument(TRUE_RHO),
            BETA_KEY: Argument(TRUE_BETA),
        },
    )

    return LorenzDataModule(
        hp=hp,
        gen_props=gen_props,
        noise_std=NOISE_STD,
        validation=validation_synthetic,
        callbacks=[DataScaling(y_scale=[1 / SCALE, 1 / SCALE, 1 / SCALE])],
    )


# --- VARIANT: source/csv ---
def create_data_module_csv(hp: ODEHyperparameters):
    from anypinn.catalog.lorenz import LorenzDataModule

    def lorenz_unscaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
        lx, ly, lz = y
        sigma = args[SIGMA_KEY]
        rho = args[RHO_KEY]
        beta = args[BETA_KEY]
        dx = sigma(x) * (ly - lx)
        dy = lx * (rho(x) - lz) - ly
        dz = lx * ly - beta(x) * lz
        return torch.stack([dx, dy, dz])

    gen_props = ODEProperties(
        ode=lorenz_unscaled,
        y0=torch.tensor([X0, Y0, Z0]),
        args={
            SIGMA_KEY: Argument(TRUE_SIGMA),
            RHO_KEY: Argument(TRUE_RHO),
            BETA_KEY: Argument(TRUE_BETA),
        },
    )

    return LorenzDataModule(
        hp=hp,
        gen_props=gen_props,
        noise_std=NOISE_STD,
        validation=validation_csv,
        callbacks=[DataScaling(y_scale=[1 / SCALE, 1 / SCALE, 1 / SCALE])],
    )


# --- END VARIANT ---

# ============================================================================
# Problem Factory
# ============================================================================


def create_problem(hp: ODEHyperparameters) -> ODEInverseProblem:
    props = ODEProperties(
        ode=lorenz_scaled,
        y0=torch.tensor([X0, Y0, Z0]) / SCALE,
        args={},
    )

    fields = FieldsRegistry(
        {
            X_KEY: Field(config=hp.fields_config),
            Y_KEY: Field(config=hp.fields_config),
            Z_KEY: Field(config=hp.fields_config),
        }
    )
    params = ParamsRegistry(
        {
            SIGMA_KEY: Parameter(config=hp.params_config),
            RHO_KEY: Parameter(config=hp.params_config),
            BETA_KEY: Parameter(config=hp.params_config),
        }
    )

    def predict_data(x_data: Tensor, fields: FieldsRegistry, _params: ParamsRegistry) -> Tensor:
        x_pred = fields[X_KEY](x_data)
        y_pred = fields[Y_KEY](x_data)
        z_pred = fields[Z_KEY](x_data)
        return torch.stack([x_pred, y_pred, z_pred], dim=1)

    return ODEInverseProblem(
        props=props,
        hp=hp,
        fields=fields,
        params=params,
        predict_data=predict_data,
    )


# ============================================================================
# Plotting and Saving
# ============================================================================


_DARK = ["#1f77b4", "#ff7f0e", "#2ca02c"]
_LIGHT = ["#aec7e8", "#ffbb78", "#98df8a"]


def plot_and_save(
    predictions: Predictions,
    results_dir: Path,
    experiment_name: str,
) -> None:
    batch, preds, trues = predictions
    t_data, y_data = batch

    x_pred = SCALE * preds[X_KEY]
    y_pred = SCALE * preds[Y_KEY]
    z_pred = SCALE * preds[Z_KEY]
    x_obs = SCALE * y_data[:, 0]
    y_obs = SCALE * y_data[:, 1]
    z_obs = SCALE * y_data[:, 2]

    sigma_pred = preds[SIGMA_KEY]
    rho_pred = preds[RHO_KEY]
    beta_pred = preds[BETA_KEY]
    sigma_true = trues[SIGMA_KEY] if trues else None
    rho_true = trues[RHO_KEY] if trues else None
    beta_true = trues[BETA_KEY] if trues else None

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Lorenz System", fontsize=14)

    ax = axes[0]
    sns.lineplot(x=t_data, y=x_pred, label=r"$x_{\mathrm{pred}}$", ax=ax, color=_DARK[0])
    sns.lineplot(x=t_data, y=y_pred, label=r"$y_{\mathrm{pred}}$", ax=ax, color=_DARK[1])
    sns.lineplot(x=t_data, y=z_pred, label=r"$z_{\mathrm{pred}}$", ax=ax, color=_DARK[2])
    sns.scatterplot(
        x=t_data, y=x_obs, label=r"$x_{\mathrm{obs}}$", ax=ax, color=_LIGHT[0], s=10, alpha=0.4
    )
    sns.scatterplot(
        x=t_data, y=y_obs, label=r"$y_{\mathrm{obs}}$", ax=ax, color=_LIGHT[1], s=10, alpha=0.4
    )
    sns.scatterplot(
        x=t_data, y=z_obs, label=r"$z_{\mathrm{obs}}$", ax=ax, color=_LIGHT[2], s=10, alpha=0.4
    )
    ax.set_title("State Predictions")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("State")
    ax.legend()

    ax = axes[1]
    if sigma_true is not None:
        sns.lineplot(
            x=t_data, y=sigma_true, label=r"$\sigma_{\mathrm{true}}$", ax=ax, color=_DARK[0]
        )
    sns.lineplot(
        x=t_data,
        y=sigma_pred,
        label=r"$\sigma_{\mathrm{pred}}$",
        linestyle="--" if sigma_true is not None else "-",
        ax=ax,
        color=_LIGHT[0] if sigma_true is not None else _DARK[0],
    )
    if rho_true is not None:
        sns.lineplot(x=t_data, y=rho_true, label=r"$\rho_{\mathrm{true}}$", ax=ax, color=_DARK[1])
    sns.lineplot(
        x=t_data,
        y=rho_pred,
        label=r"$\rho_{\mathrm{pred}}$",
        linestyle="--" if rho_true is not None else "-",
        ax=ax,
        color=_LIGHT[1] if rho_true is not None else _DARK[1],
    )
    if beta_true is not None:
        sns.lineplot(
            x=t_data, y=beta_true, label=r"$\beta_{\mathrm{true}}$", ax=ax, color=_DARK[2]
        )
    sns.lineplot(
        x=t_data,
        y=beta_pred,
        label=r"$\beta_{\mathrm{pred}}$",
        linestyle="--" if beta_true is not None else "-",
        ax=ax,
        color=_LIGHT[2] if beta_true is not None else _DARK[2],
    )
    ax.set_title("Parameter Recovery")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Value")
    top = ax.get_ylim()[1]
    pad = top * 0.10
    ax.set_ylim(-pad, top + pad)
    ax.legend()

    plt.tight_layout()
    fig.savefig(results_dir / "plot.png", dpi=300)
    plt.close(fig)

    df = pd.DataFrame(
        {
            "t": t_data,
            "x_obs": x_obs,
            "y_obs": y_obs,
            "z_obs": z_obs,
            "x_pred": x_pred,
            "y_pred": y_pred,
            "z_pred": z_pred,
            "sigma_pred": sigma_pred,
            "rho_pred": rho_pred,
            "beta_pred": beta_pred,
            "sigma_true": sigma_true,
            "rho_true": rho_true,
            "beta_true": beta_true,
        }
    )
    df.to_csv(results_dir / "data.csv", index=False, float_format="%.6e")
