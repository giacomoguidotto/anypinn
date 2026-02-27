from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import Tensor

from anypinn.catalog.lorenz import (
    BETA_KEY,
    RHO_KEY,
    SIGMA_KEY,
    X_KEY,
    Y_KEY,
    Z_KEY,
    LorenzDataModule,
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

TRUE_SIGMA = 10.0
TRUE_RHO = 28.0
TRUE_BETA = 8.0 / 3.0

X0 = -8.0
Y0 = 7.0
Z0 = 27.0

T_TOTAL = 3
NOISE_STD = 0.5
SCALE = 20.0


# ============================================================================
# ODE Functions
# ============================================================================


def lorenz_unscaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    """3-state Lorenz system for odeint data generation (physical units)."""
    lx, ly, lz = y
    sigma = args[SIGMA_KEY]
    rho = args[RHO_KEY]
    beta = args[BETA_KEY]

    dx = sigma(x) * (ly - lx)
    dy = lx * (rho(x) - lz) - ly
    dz = lx * ly - beta(x) * lz
    return torch.stack([dx, dy, dz])


def lorenz_scaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    """Scaled Lorenz ODE for training. States pre-divided by SCALE, time by T_TOTAL."""
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
# Data and Problem Factories
# ============================================================================

validation: ValidationRegistry = {
    SIGMA_KEY: lambda x: torch.full_like(x, TRUE_SIGMA),
    RHO_KEY: lambda x: torch.full_like(x, TRUE_RHO),
    BETA_KEY: lambda x: torch.full_like(x, TRUE_BETA),
}

gen_props = ODEProperties(
    ode=lorenz_unscaled,
    y0=torch.tensor([X0, Y0, Z0]),
    args={
        SIGMA_KEY: Argument(TRUE_SIGMA),
        RHO_KEY: Argument(TRUE_RHO),
        BETA_KEY: Argument(TRUE_BETA),
    },
)


def create_data_module(hp: ODEHyperparameters) -> LorenzDataModule:
    return LorenzDataModule(
        hp=hp,
        gen_props=gen_props,
        noise_std=NOISE_STD,
        validation=validation,
        callbacks=[DataScaling(y_scale=[1 / SCALE, 1 / SCALE, 1 / SCALE])],
    )


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
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Subplot 1: x, y, z predictions vs data
    ax = axes[0]
    sns.lineplot(x=t_data, y=x_pred, label="$x_{pred}$", ax=ax, color="C0")
    sns.lineplot(x=t_data, y=y_pred, label="$y_{pred}$", ax=ax, color="C1")
    sns.lineplot(x=t_data, y=z_pred, label="$z_{pred}$", ax=ax, color="C2")
    sns.scatterplot(x=t_data, y=x_obs, label="$x_{obs}$", ax=ax, color="C0", s=10, alpha=0.3)
    sns.scatterplot(x=t_data, y=y_obs, label="$y_{obs}$", ax=ax, color="C1", s=10, alpha=0.3)
    sns.scatterplot(x=t_data, y=z_obs, label="$z_{obs}$", ax=ax, color="C2", s=10, alpha=0.3)
    ax.set_title("Lorenz System Predictions")
    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    ax.legend()

    # Subplot 2: sigma prediction
    ax = axes[1]
    if sigma_true is not None:
        sns.lineplot(x=t_data, y=sigma_true, label=r"$\sigma_{true}$", ax=ax, color="C0")
    sns.lineplot(
        x=t_data,
        y=sigma_pred,
        label=r"$\sigma_{pred}$",
        linestyle="--" if sigma_true is not None else "-",
        ax=ax,
        color="C3" if sigma_true is not None else "C0",
    )
    ax.set_title(r"$\sigma$ Prediction")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\sigma$")
    ax.legend()

    # Subplot 3: rho and beta predictions
    ax = axes[2]
    if rho_true is not None:
        sns.lineplot(x=t_data, y=rho_true, label=r"$\rho_{true}$", ax=ax, color="C0")
    sns.lineplot(
        x=t_data,
        y=rho_pred,
        label=r"$\rho_{pred}$",
        linestyle="--" if rho_true is not None else "-",
        ax=ax,
        color="C3" if rho_true is not None else "C0",
    )
    if beta_true is not None:
        sns.lineplot(x=t_data, y=beta_true, label=r"$\beta_{true}$", ax=ax, color="C1")
    sns.lineplot(
        x=t_data,
        y=beta_pred,
        label=r"$\beta_{pred}$",
        linestyle="--" if beta_true is not None else "-",
        ax=ax,
        color="C4" if beta_true is not None else "C1",
    )
    ax.set_title(r"$\rho$, $\beta$ Predictions")
    ax.set_xlabel("Time")
    ax.set_ylabel("Parameter value")
    ax.legend()

    plt.tight_layout()
    fig.savefig(results_dir / f"{experiment_name}.png", dpi=300)
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
    df.to_csv(results_dir / f"{experiment_name}.csv", index=False, float_format="%.6e")
