from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import Tensor

from anypinn.catalog.damped_oscillator import (
    OMEGA_KEY,
    V_KEY,
    X_KEY,
    ZETA_KEY,
    DampedOscillatorDataModule,
)
from anypinn.core import (
    ArgsRegistry,
    Argument,
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
T = T_TOTAL

# Noise level for synthetic data
NOISE_STD = 0.02


# ============================================================================
# ODE Functions
# ============================================================================


def oscillator_unscaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    pos, vel = y
    z = args[ZETA_KEY]
    omega0 = args[OMEGA_KEY]
    dx = vel
    dv = -2 * z(x) * omega0(x) * vel - omega0(x) ** 2 * pos
    return torch.stack([dx, dv])


def oscillator_scaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    pos, vel = y
    z = args[ZETA_KEY]
    omega0 = args[OMEGA_KEY]

    dx = vel
    dv = -2 * z(x) * omega0(x) * vel - omega0(x) ** 2 * pos

    dx = dx * T
    dv = dv * T
    return torch.stack([dx, dv])


# ============================================================================
# Data and Problem Factories
# ============================================================================

validation: ValidationRegistry = {
    ZETA_KEY: lambda x: torch.full_like(x, TRUE_ZETA),
}

gen_props = ODEProperties(
    ode=oscillator_unscaled,
    y0=torch.tensor([X0, V0]),
    args={
        ZETA_KEY: Argument(TRUE_ZETA),
        OMEGA_KEY: Argument(TRUE_OMEGA0),
    },
)


def create_data_module(hp: ODEHyperparameters) -> DampedOscillatorDataModule:
    return DampedOscillatorDataModule(
        hp=hp,
        gen_props=gen_props,
        noise_std=NOISE_STD,
        validation=validation,
        callbacks=[DataScaling(y_scale=1.0)],
    )


def create_problem(hp: ODEHyperparameters) -> ODEInverseProblem:
    props = ODEProperties(
        ode=oscillator_scaled,
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

    # plot
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: position + data, velocity predicted
    ax = axes[0]
    sns.lineplot(x=t_data, y=x_pred, label="$x_{pred}$ (position)", ax=ax, color="C0")
    sns.lineplot(x=t_data, y=v_pred, label="$v_{pred}$ (velocity)", ax=ax, color="C2")
    sns.scatterplot(x=t_data, y=x_data, label="$x_{observed}$", ax=ax, color="C1", s=10, alpha=0.5)
    ax.set_title("Damped Oscillator Predictions")
    ax.set_xlabel("Time (scaled)")
    ax.set_ylabel("Amplitude")
    ax.legend()

    # Subplot 2: zeta predicted vs true
    ax = axes[1]
    if zeta_true is not None:
        sns.lineplot(x=t_data, y=zeta_true, label=r"$\zeta_{true}$", ax=ax, color="C0")
    sns.lineplot(
        x=t_data,
        y=zeta_pred,
        label=r"$\zeta_{pred}$",
        linestyle="--" if zeta_true is not None else "-",
        ax=ax,
        color="C3" if zeta_true is not None else "C0",
    )
    ax.set_title(r"$\zeta$ (Damping Ratio) Prediction")
    ax.set_xlabel("Time (scaled)")
    ax.set_ylabel(r"$\zeta$")
    ax.legend()

    plt.tight_layout()

    fig.savefig(results_dir / f"{experiment_name}.png", dpi=300)
    plt.close(fig)

    # save
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

    df.to_csv(results_dir / f"{experiment_name}.csv", index=False, float_format="%.6e")
