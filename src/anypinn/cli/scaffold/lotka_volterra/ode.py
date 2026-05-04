"""Lotka-Volterra predator-prey — mathematical definition."""

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
ALPHA_KEY = "alpha"
BETA_KEY = "beta"
DELTA_KEY = "delta"
GAMMA_KEY = "gamma"

# ============================================================================
# Constants
# ============================================================================

# True parameter values
TRUE_ALPHA = 0.5
TRUE_BETA = 0.02
TRUE_DELTA = 0.01
TRUE_GAMMA = 0.5

# Initial conditions (populations)
X0 = 40.0
Y0 = 9.0

# Time domain
T_TOTAL = 50

# Population scaling
POP_SCALE = 100.0

# Noise level for synthetic data (std as fraction of signal)
NOISE_FRAC = 0.02

# ============================================================================
# Fourier encoding
# ============================================================================


def fourier_encode(t: Tensor) -> Tensor:
    features = [t]
    for k in range(1, 7):
        features.append(torch.sin(k * t))
    return torch.cat(features, dim=-1)


# ============================================================================
# ODE Definition
# ============================================================================


def lotka_volterra(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    """Scaled Lotka-Volterra ODE. Populations scaled by POP_SCALE, time by T_TOTAL."""
    prey, predator = y
    b = args[BETA_KEY]
    alpha = args[ALPHA_KEY]
    delta = args[DELTA_KEY]
    gamma = args[GAMMA_KEY]

    dx = alpha(x) * prey - b(x) * prey * predator * POP_SCALE
    dy = delta(x) * prey * predator * POP_SCALE - gamma(x) * predator

    dx = dx * T_TOTAL
    dy = dy * T_TOTAL
    return torch.stack([dx, dy])


# ============================================================================
# Validation
# ============================================================================

# --- VARIANT: source/synthetic ---
validation_synthetic: ValidationRegistry = {
    BETA_KEY: lambda x: torch.full_like(x, TRUE_BETA),
}
# --- VARIANT: source/csv ---
validation_csv: ValidationRegistry = {
    # TODO: map beta to a column in your CSV
    # BETA_KEY: ColumnRef(column="your_column"),
}
# --- END VARIANT ---

# ============================================================================
# Data Module Factory
# ============================================================================


# --- VARIANT: source/synthetic ---
def create_data_module_synthetic(hp: ODEHyperparameters):
    from anypinn.catalog.lotka_volterra import LotkaVolterraDataModule

    def LV_unscaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
        prey, predator = y
        b = args[BETA_KEY]
        alpha = args[ALPHA_KEY]
        delta = args[DELTA_KEY]
        gamma = args[GAMMA_KEY]
        dx = alpha(x) * prey - b(x) * prey * predator
        dy = delta(x) * prey * predator - gamma(x) * predator
        return torch.stack([dx, dy])

    gen_props = ODEProperties(
        ode=LV_unscaled,
        y0=torch.tensor([X0, Y0]),
        args={
            ALPHA_KEY: Argument(TRUE_ALPHA),
            BETA_KEY: Argument(TRUE_BETA),
            DELTA_KEY: Argument(TRUE_DELTA),
            GAMMA_KEY: Argument(TRUE_GAMMA),
        },
    )

    return LotkaVolterraDataModule(
        hp=hp,
        gen_props=gen_props,
        noise_frac=NOISE_FRAC,
        validation=validation_synthetic,
        callbacks=[DataScaling(y_scale=[1 / POP_SCALE, 1 / POP_SCALE])],
    )


# --- VARIANT: source/csv ---
def create_data_module_csv(hp: ODEHyperparameters):
    from anypinn.catalog.lotka_volterra import LotkaVolterraDataModule

    return LotkaVolterraDataModule(
        hp=hp,
        validation=validation_csv,
        callbacks=[DataScaling(y_scale=[1 / POP_SCALE, 1 / POP_SCALE])],
    )


# --- END VARIANT ---

# ============================================================================
# Problem Factory
# ============================================================================


def create_problem(hp: ODEHyperparameters) -> ODEInverseProblem:
    props = ODEProperties(
        ode=lotka_volterra,
        y0=torch.tensor([X0, Y0]) / POP_SCALE,
        args={
            ALPHA_KEY: Argument(TRUE_ALPHA),
            DELTA_KEY: Argument(TRUE_DELTA),
            GAMMA_KEY: Argument(TRUE_GAMMA),
        },
    )

    fields = FieldsRegistry(
        {
            X_KEY: Field(config=hp.fields_config),
            Y_KEY: Field(config=hp.fields_config),
        }
    )
    params = ParamsRegistry(
        {
            BETA_KEY: Parameter(config=hp.params_config),
        }
    )

    def predict_data(x_data: Tensor, fields: FieldsRegistry, _params: ParamsRegistry) -> Tensor:
        x_pred = fields[X_KEY](x_data)
        y_pred = fields[Y_KEY](x_data)
        return torch.stack([x_pred, y_pred], dim=1)

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


_DARK = ["#1f77b4", "#ff7f0e"]
_LIGHT = ["#aec7e8", "#ffbb78"]


def plot_and_save(
    predictions: Predictions,
    results_dir: Path,
    experiment_name: str,
) -> None:
    batch, preds, trues = predictions
    t_data, y_data = batch

    x_pred = POP_SCALE * preds[X_KEY]
    y_pred = POP_SCALE * preds[Y_KEY]

    x_obs = POP_SCALE * y_data[:, 0]
    y_obs = POP_SCALE * y_data[:, 1]

    beta_pred = preds[BETA_KEY]
    beta_true = trues[BETA_KEY] if trues else None

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Lotka--Volterra System", fontsize=14)

    ax = axes[0]
    sns.lineplot(x=t_data, y=x_pred, label=r"$x_{\mathrm{pred}}$", ax=ax, color=_DARK[0])
    sns.lineplot(x=t_data, y=y_pred, label=r"$y_{\mathrm{pred}}$", ax=ax, color=_DARK[1])
    sns.scatterplot(
        x=t_data, y=x_obs, label=r"$x_{\mathrm{obs}}$", ax=ax, color=_LIGHT[0], s=10, alpha=0.4
    )
    sns.scatterplot(
        x=t_data, y=y_obs, label=r"$y_{\mathrm{obs}}$", ax=ax, color=_LIGHT[1], s=10, alpha=0.4
    )
    ax.set_title("State Predictions")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Population")
    ax.legend()

    ax = axes[1]
    if beta_true is not None:
        sns.lineplot(
            x=t_data, y=beta_true, label=r"$\beta_{\mathrm{true}}$", ax=ax, color=_DARK[0]
        )
    sns.lineplot(
        x=t_data,
        y=beta_pred,
        label=r"$\beta_{\mathrm{pred}}$",
        linestyle="--" if beta_true is not None else "-",
        ax=ax,
        color=_LIGHT[0] if beta_true is not None else _DARK[0],
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
            "x_pred": x_pred,
            "y_pred": y_pred,
            "beta_pred": beta_pred,
            "beta_true": beta_true,
        }
    )
    df.to_csv(results_dir / "data.csv", index=False, float_format="%.6e")
