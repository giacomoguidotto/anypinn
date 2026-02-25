from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import Tensor

from anypinn.catalog.seir import (
    BETA_KEY,
    E_KEY,
    GAMMA_KEY,
    I_KEY,
    S_KEY,
    SIGMA_KEY,
    SEIRDataModule,
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
TRUE_BETA = 0.5
TRUE_SIGMA = 1 / 5.2
TRUE_GAMMA = 1 / 10

# Initial conditions (fractions)
S0 = 0.99
E0 = 0.01
I0 = 0.001

# Time domain
T_DAYS = 160
T = T_DAYS

# Noise level for synthetic data
NOISE_STD = 0.0005


# ============================================================================
# ODE Functions
# ============================================================================


def SEIR_unscaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    S, E, I = y
    b = args[BETA_KEY]
    sigma = args[SIGMA_KEY]
    gamma = args[GAMMA_KEY]
    dS = -b(x) * S * I
    dE = b(x) * S * I - sigma(x) * E
    dI = sigma(x) * E - gamma(x) * I
    return torch.stack([dS, dE, dI])


def SEIR_scaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    S, E, I = y
    b = args[BETA_KEY]
    sigma = args[SIGMA_KEY]
    gamma = args[GAMMA_KEY]

    dS = -b(x) * S * I
    dE = b(x) * S * I - sigma(x) * E
    dI = sigma(x) * E - gamma(x) * I

    dS = dS * T
    dE = dE * T
    dI = dI * T
    return torch.stack([dS, dE, dI])


# ============================================================================
# Data and Problem Factories
# ============================================================================

validation: ValidationRegistry = {
    BETA_KEY: lambda x: torch.full_like(x, TRUE_BETA),
}

gen_props = ODEProperties(
    ode=SEIR_unscaled,
    y0=torch.tensor([S0, E0, I0]),
    args={
        BETA_KEY: Argument(TRUE_BETA),
        SIGMA_KEY: Argument(TRUE_SIGMA),
        GAMMA_KEY: Argument(TRUE_GAMMA),
    },
)


def create_data_module(hp: ODEHyperparameters) -> SEIRDataModule:
    return SEIRDataModule(
        hp=hp,
        gen_props=gen_props,
        noise_std=NOISE_STD,
        validation=validation,
        callbacks=[DataScaling(y_scale=1.0)],
    )


def create_problem(hp: ODEHyperparameters) -> ODEInverseProblem:
    props = ODEProperties(
        ode=SEIR_scaled,
        y0=torch.tensor([S0, E0, I0]),
        args={
            SIGMA_KEY: Argument(TRUE_SIGMA),
            GAMMA_KEY: Argument(TRUE_GAMMA),
        },
    )

    fields = FieldsRegistry(
        {
            S_KEY: Field(config=hp.fields_config),
            E_KEY: Field(config=hp.fields_config),
            I_KEY: Field(config=hp.fields_config),
        }
    )
    params = ParamsRegistry(
        {
            BETA_KEY: Parameter(config=hp.params_config),
        }
    )

    def predict_data(x_data: Tensor, fields: FieldsRegistry, _params: ParamsRegistry) -> Tensor:
        I_pred = fields[I_KEY](x_data)
        return I_pred.unsqueeze(1)

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
    t_data, I_data = batch
    I_data = I_data.squeeze(-1)

    S_pred = preds[S_KEY]
    E_pred = preds[E_KEY]
    I_pred = preds[I_KEY]

    beta_pred = preds[BETA_KEY]
    beta_true = trues[BETA_KEY] if trues else None

    # plot
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: S, E, I curves + observed I data
    ax = axes[0]
    sns.lineplot(x=t_data, y=S_pred, label="$S_{pred}$", ax=ax, color="C0")
    sns.lineplot(x=t_data, y=E_pred, label="$E_{pred}$", ax=ax, color="C2")
    sns.lineplot(x=t_data, y=I_pred, label="$I_{pred}$", ax=ax, color="C3")
    sns.scatterplot(x=t_data, y=I_data, label="$I_{observed}$", ax=ax, color="C1", s=10, alpha=0.5)
    ax.set_title("SEIR Model Predictions")
    ax.set_xlabel("Time (scaled)")
    ax.set_ylabel("Fraction")
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
            "I_observed": I_data,
            "S_pred": S_pred,
            "E_pred": E_pred,
            "I_pred": I_pred,
            "beta_pred": beta_pred,
            "beta_true": beta_true,
        }
    )

    df.to_csv(results_dir / f"{experiment_name}.csv", index=False, float_format="%.6e")
