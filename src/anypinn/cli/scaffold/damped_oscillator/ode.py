"""Damped oscillator — mathematical definition."""

from __future__ import annotations

import math
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
V_KEY = "v"
ZETA_KEY = "zeta"
OMEGA_KEY = "omega0"

# ============================================================================
# Constants
# ============================================================================

# True parameter values
TRUE_ZETA = 0.15
TRUE_OMEGA0 = 2 * math.pi

# Initial conditions
X0 = 1.0
V0 = 0.0

# Time domain (seconds)
T_TOTAL = 5

# Noise level for synthetic data
NOISE_STD = 0.02

# ============================================================================
# ODE Definition
# ============================================================================


def oscillator(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    """Scaled damped oscillator ODE: $dx/dt = v$, $dv/dt = -2 zeta omega_0 v - omega_0^2 x$."""
    pos, vel = y
    z = args[ZETA_KEY]
    omega0 = args[OMEGA_KEY]

    dx = vel
    dv = -2 * z(x) * omega0(x) * vel - omega0(x) ** 2 * pos

    dx = dx * T_TOTAL
    dv = dv * T_TOTAL
    return torch.stack([dx, dv])


# ============================================================================
# Validation
# ============================================================================

# --- VARIANT: source/synthetic ---
validation_synthetic: ValidationRegistry = {
    ZETA_KEY: lambda x: torch.full_like(x, TRUE_ZETA),
}
# --- VARIANT: source/csv ---
validation_csv: ValidationRegistry = {
    # TODO: map zeta to a column in your CSV
    # ZETA_KEY: ColumnRef(column="your_column"),
}
# --- END VARIANT ---

# ============================================================================
# Data Module Factory
# ============================================================================


# --- VARIANT: source/synthetic ---
def create_data_module_synthetic(hp: ODEHyperparameters):
    from anypinn.catalog.damped_oscillator import DampedOscillatorDataModule

    def oscillator_unscaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
        pos, vel = y
        z = args[ZETA_KEY]
        omega0 = args[OMEGA_KEY]
        dx = vel
        dv = -2 * z(x) * omega0(x) * vel - omega0(x) ** 2 * pos
        return torch.stack([dx, dv])

    gen_props = ODEProperties(
        ode=oscillator_unscaled,
        y0=torch.tensor([X0, V0]),
        args={
            ZETA_KEY: Argument(TRUE_ZETA),
            OMEGA_KEY: Argument(TRUE_OMEGA0),
        },
    )

    return DampedOscillatorDataModule(
        hp=hp,
        gen_props=gen_props,
        noise_std=NOISE_STD,
        validation=validation_synthetic,
        callbacks=[DataScaling(y_scale=1.0)],
    )


# --- VARIANT: source/csv ---
def create_data_module_csv(hp: ODEHyperparameters):
    from anypinn.catalog.damped_oscillator import DampedOscillatorDataModule

    def oscillator_unscaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
        pos, vel = y
        z = args[ZETA_KEY]
        omega0 = args[OMEGA_KEY]
        dx = vel
        dv = -2 * z(x) * omega0(x) * vel - omega0(x) ** 2 * pos
        return torch.stack([dx, dv])

    gen_props = ODEProperties(
        ode=oscillator_unscaled,
        y0=torch.tensor([X0, V0]),
        args={
            ZETA_KEY: Argument(TRUE_ZETA),
            OMEGA_KEY: Argument(TRUE_OMEGA0),
        },
    )

    return DampedOscillatorDataModule(
        hp=hp,
        gen_props=gen_props,
        noise_std=NOISE_STD,
        validation=validation_csv,
        callbacks=[DataScaling(y_scale=1.0)],
    )


# --- END VARIANT ---

# ============================================================================
# Problem Factory
# ============================================================================


def create_problem(hp: ODEHyperparameters) -> ODEInverseProblem:
    props = ODEProperties(
        ode=oscillator,
        y0=torch.tensor([X0, V0]),
        args={
            OMEGA_KEY: Argument(TRUE_OMEGA0),
        },
    )

    fields = FieldsRegistry(
        {
            X_KEY: Field(config=hp.fields_config),
            V_KEY: Field(config=hp.fields_config),
        }
    )
    params = ParamsRegistry(
        {
            ZETA_KEY: Parameter(config=hp.params_config),
        }
    )

    def predict_data(x_data: Tensor, fields: FieldsRegistry, _params: ParamsRegistry) -> Tensor:
        x_pred = fields[X_KEY](x_data)
        return x_pred.unsqueeze(1)

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
    t_data, x_data = batch
    x_data = x_data.squeeze(-1)

    x_pred = preds[X_KEY]
    v_pred = preds[V_KEY]

    zeta_pred = preds[ZETA_KEY]
    zeta_true = trues[ZETA_KEY] if trues else None

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Damped Harmonic Oscillator", fontsize=14)

    ax = axes[0]
    sns.lineplot(x=t_data, y=x_pred, label=r"$x_{\mathrm{pred}}$", ax=ax, color=_DARK[0])
    sns.lineplot(x=t_data, y=v_pred, label=r"$v_{\mathrm{pred}}$", ax=ax, color=_DARK[1])
    sns.scatterplot(
        x=t_data, y=x_data, label=r"$x_{\mathrm{obs}}$", ax=ax, color=_LIGHT[0], s=10, alpha=0.4
    )
    ax.set_title("State Predictions")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Amplitude")
    ax.legend()

    ax = axes[1]
    if zeta_true is not None:
        sns.lineplot(
            x=t_data, y=zeta_true, label=r"$\zeta_{\mathrm{true}}$", ax=ax, color=_DARK[0]
        )
    sns.lineplot(
        x=t_data,
        y=zeta_pred,
        label=r"$\zeta_{\mathrm{pred}}$",
        linestyle="--" if zeta_true is not None else "-",
        ax=ax,
        color=_LIGHT[0] if zeta_true is not None else _DARK[0],
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
            "x_observed": x_data,
            "x_pred": x_pred,
            "v_pred": v_pred,
            "zeta_pred": zeta_pred,
            "zeta_true": zeta_true,
        }
    )
    df.to_csv(results_dir / "data.csv", index=False, float_format="%.6e")
