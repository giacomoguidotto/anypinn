from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import Tensor

from anypinn.catalog.lotka_volterra import (
    ALPHA_KEY,
    BETA_KEY,
    DELTA_KEY,
    GAMMA_KEY,
    X_KEY,
    Y_KEY,
    LotkaVolterraDataModule,
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
TRUE_ALPHA = 0.5
TRUE_BETA = 0.02
TRUE_DELTA = 0.01
TRUE_GAMMA = 0.5

# Initial conditions (populations)
X0 = 40.0
Y0 = 9.0

# Time domain
T_TOTAL = 50
T = T_TOTAL

# Population scaling
POP_SCALE = 100.0

# Noise level for synthetic data (std as fraction of signal)
NOISE_FRAC = 0.02


# ============================================================================
# Fourier Encoding
# ============================================================================


def fourier_encode(t: Tensor) -> Tensor:
    features = [t]
    for k in range(1, 7):
        features.append(torch.sin(k * t))
    return torch.cat(features, dim=-1)


# ============================================================================
# ODE Functions
# ============================================================================


def LV_unscaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    prey, predator = y
    b = args[BETA_KEY]
    alpha = args[ALPHA_KEY]
    delta = args[DELTA_KEY]
    gamma = args[GAMMA_KEY]
    dx = alpha(x) * prey - b(x) * prey * predator
    dy = delta(x) * prey * predator - gamma(x) * predator
    return torch.stack([dx, dy])


def LV_scaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    prey, predator = y
    b = args[BETA_KEY]
    alpha = args[ALPHA_KEY]
    delta = args[DELTA_KEY]
    gamma = args[GAMMA_KEY]

    dx = alpha(x) * prey - b(x) * prey * predator * POP_SCALE
    dy = delta(x) * prey * predator * POP_SCALE - gamma(x) * predator

    dx = dx * T
    dy = dy * T
    return torch.stack([dx, dy])


# ============================================================================
# Data and Problem Factories
# ============================================================================

validation: ValidationRegistry = {
    BETA_KEY: lambda x: torch.full_like(x, TRUE_BETA),
}

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


def create_data_module(hp: ODEHyperparameters) -> LotkaVolterraDataModule:
    return LotkaVolterraDataModule(
        hp=hp,
        gen_props=gen_props,
        noise_frac=NOISE_FRAC,
        validation=validation,
        callbacks=[DataScaling(y_scale=[1 / POP_SCALE, 1 / POP_SCALE])],
    )


def create_problem(hp: ODEHyperparameters) -> ODEInverseProblem:
    props = ODEProperties(
        ode=LV_scaled,
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


def plot_and_save(
    predictions: Predictions,
    results_dir: Path,
    experiment_name: str,
) -> None:
    batch, preds, trues = predictions
    t_data, y_data = batch

    # Unscale predictions back to original population scale
    x_pred = POP_SCALE * preds[X_KEY]
    y_pred = POP_SCALE * preds[Y_KEY]

    # Unscale observed data
    x_obs = POP_SCALE * y_data[:, 0]
    y_obs = POP_SCALE * y_data[:, 1]

    beta_pred = preds[BETA_KEY]
    beta_true = trues[BETA_KEY] if trues else None

    # plot
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: prey and predator curves + observed data
    ax = axes[0]
    sns.lineplot(x=t_data, y=x_pred, label="Prey $x_{pred}$", ax=ax, color="C0")
    sns.lineplot(x=t_data, y=y_pred, label="Predator $y_{pred}$", ax=ax, color="C3")
    sns.scatterplot(x=t_data, y=x_obs, label="Prey $x_{obs}$", ax=ax, color="C0", s=10, alpha=0.4)
    sns.scatterplot(
        x=t_data, y=y_obs, label="Predator $y_{obs}$", ax=ax, color="C3", s=10, alpha=0.4
    )
    ax.set_title("Lotka-Volterra Predictions")
    ax.set_xlabel("Time (scaled)")
    ax.set_ylabel("Population")
    ax.legend()

    # Subplot 2: beta predicted vs true
    ax = axes[1]
    if beta_true is not None:
        sns.lineplot(x=t_data, y=beta_true, label=r"$\beta_{true}$", ax=ax, color="C0")
    sns.lineplot(
        x=t_data,
        y=beta_pred,
        label=r"$\beta_{pred}$",
        linestyle="--" if beta_true is not None else "-",
        ax=ax,
        color="C3" if beta_true is not None else "C0",
    )
    ax.set_title(r"$\beta$ Parameter Prediction")
    ax.set_xlabel("Time (scaled)")
    ax.set_ylabel(r"$\beta$")
    ax.legend()

    plt.tight_layout()

    fig.savefig(results_dir / f"{experiment_name}.png", dpi=300)
    plt.close(fig)

    # save
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

    df.to_csv(results_dir / f"{experiment_name}.csv", index=False, float_format="%.6e")
