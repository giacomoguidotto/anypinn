from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import Tensor

from anypinn.catalog.sir import DELTA_KEY, I_KEY, Rt_KEY, SIRInvDataModule
from anypinn.core import (
    ArgsRegistry,
    Argument,
    ColumnRef,
    Field,
    FieldsRegistry,
    MLPConfig,
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

SIGMA_KEY = "sigma"
H_KEY = "H"

C_I = 1e6
C_H = 1e3
T = 120


# ============================================================================
# ODE Function
# ============================================================================


def hSIR_s(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    """Reduced SIR ODE with hospitalization constraint: dI/dt = δ(Rt - 1)I"""
    I = y
    d, Rt = args[DELTA_KEY], args[Rt_KEY]

    dI = d(x) * (Rt(x) - 1) * I
    dI = dI * T
    return dI


# ============================================================================
# Problem Definition (module-level props for use in plot_and_save)
# ============================================================================

props = ODEProperties(
    ode=hSIR_s,
    y0=torch.tensor([1]) / C_I,
    args={
        DELTA_KEY: Argument(1 / 5),
    },
)


# ============================================================================
# Data and Problem Factories
# ============================================================================


def create_data_module(hp: ODEHyperparameters) -> SIRInvDataModule:
    validation: ValidationRegistry = {
        Rt_KEY: ColumnRef(column="Rt"),
        SIGMA_KEY: ColumnRef(column="sigma"),
    }
    return SIRInvDataModule(
        hp=hp,
        validation=validation,
        callbacks=[DataScaling(y_scale=[1 / C_I, 1 / C_H])],
    )


def create_problem(hp: ODEHyperparameters) -> ODEInverseProblem:
    fields = FieldsRegistry(
        {
            I_KEY: Field(config=hp.fields_config),
        }
    )
    params = ParamsRegistry(
        {
            Rt_KEY: Parameter(config=hp.params_config),
            SIGMA_KEY: Parameter(
                config=MLPConfig(
                    in_dim=1,
                    out_dim=1,
                    hidden_layers=10 * [5],
                    activation="tanh",
                    output_activation="softplus",
                )
            ),
        }
    )

    def predict_data(x_data: Tensor, fields: FieldsRegistry, params: ParamsRegistry) -> Tensor:
        delta = props.args[DELTA_KEY]
        I = fields[I_KEY]
        sigma = params[SIGMA_KEY]

        I_pred: Tensor = I(x_data)
        H_pred: Tensor = (delta(x_data) * C_I * sigma(x_data) * I_pred) / C_H

        return torch.stack([I_pred, H_pred], dim=1)  # [n, 2, 1]

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
    props: ODEProperties,
    C_I: float,
    C_H: float,
) -> None:
    batch, preds, trues = predictions
    t_data, y_data = batch
    I_obs_data, H_obs_data = y_data[:, 0], y_data[:, 1]
    delta = props.args[DELTA_KEY](t_data)

    I_pred = C_I * preds[I_KEY]
    Rt_pred = preds[Rt_KEY]
    sigma_pred = preds[SIGMA_KEY]

    H_pred = delta * sigma_pred * I_pred

    I_obs = C_I * I_obs_data
    H_obs = C_H * H_obs_data

    Rt_true = trues[Rt_KEY] if trues and Rt_KEY in trues else None
    sigma_true = trues[SIGMA_KEY] if trues and SIGMA_KEY in trues else None

    # plot
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax = axes[0, 0]
    sns.lineplot(x=t_data, y=H_pred, label=r"$\Delta H_{pred}$", ax=ax, color="C0")
    sns.scatterplot(
        x=t_data, y=H_obs, label=r"$\Delta H_{obs}$", ax=ax, color="C1", s=20, alpha=0.6
    )
    ax.set_title("Daily Hospitalizations")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Daily Hospitalizations")
    ax.legend()

    ax = axes[0, 1]
    sns.lineplot(x=t_data, y=I_pred, label="$I_{pred}$", ax=ax, color="C2")
    sns.scatterplot(x=t_data, y=I_obs, label="$I_{obs}$", ax=ax, color="C1", s=20, alpha=0.6)
    ax.set_title("Predicted Infected Population")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("I (Population)")
    ax.legend()

    ax = axes[1, 0]
    if Rt_true is not None:
        sns.lineplot(x=t_data, y=Rt_true, label=r"$R_{t, true}$", ax=ax, color="C3")
    sns.lineplot(
        x=t_data,
        y=Rt_pred,
        label=r"$R_{t, pred}$",
        ax=ax,
        linestyle="--" if Rt_true is not None else "-",
        color="C4" if Rt_true is not None else "C3",
    )
    ax.axhline(y=1, color="red", linestyle=":", alpha=0.5, label="$R_t = 1$")
    ax.set_title(r"$R_t$ (Reproduction Number)")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel(r"$R_t$")
    ax.legend()

    ax = axes[1, 1]
    if sigma_true is not None:
        sns.lineplot(x=t_data, y=sigma_true, label=r"$\sigma_{true}$", ax=ax, color="C5")
    sns.lineplot(
        x=t_data,
        y=sigma_pred,
        label=r"$\sigma_{pred}$",
        ax=ax,
        linestyle="--" if sigma_true is not None else "-",
        color="C6" if sigma_true is not None else "C5",
    )
    ax.set_title(r"$\sigma$ (Hospitalization Rate)")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel(r"$\sigma$ (fraction)")
    ax.legend()

    plt.tight_layout()

    fig.savefig(results_dir / f"{experiment_name}.png", dpi=300)
    plt.close(fig)

    # save
    df = pd.DataFrame(
        {
            "t": t_data,
            "H_obs": H_obs,
            "H_pred": H_pred,
            "I_pred": I_pred,
            "Rt_pred": Rt_pred,
            "sigma_pred": sigma_pred,
        }
    )

    if Rt_true is not None:
        df["Rt_true"] = Rt_true
    if sigma_true is not None:
        df["sigma_true"] = sigma_true

    df.to_csv(results_dir / f"{experiment_name}.csv", index=False, float_format="%.6e")
