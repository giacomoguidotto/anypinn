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
START_DATE = "2020-02-01"
N_DAYS = 118


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


_DARK = ["#1f77b4", "#ff7f0e", "#2ca02c"]
_LIGHT = ["#aec7e8", "#ffbb78", "#98df8a"]


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

    dates = pd.to_datetime(START_DATE) + pd.to_timedelta(t_data * N_DAYS, unit="D")

    I_pred = C_I * preds[I_KEY]
    Rt_pred = preds[Rt_KEY]
    sigma_pred = preds[SIGMA_KEY]

    H_pred = delta * sigma_pred * I_pred

    I_obs = C_I * I_obs_data
    H_obs = C_H * H_obs_data

    Rt_true = trues[Rt_KEY] if trues and Rt_KEY in trues else None
    sigma_true = trues[SIGMA_KEY] if trues and SIGMA_KEY in trues else None

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Hospitalized SIR Model", fontsize=14)

    ax = axes[0, 0]
    ax.plot(dates, H_pred, label=r"$\Delta H_{\mathrm{pred}}$", color=_DARK[0])
    ax.scatter(dates, H_obs, label=r"$\Delta H_{\mathrm{obs}}$", color=_LIGHT[0], s=10, alpha=0.4)
    ax.set_title("State Predictions")
    ax.set_ylabel("Daily Hospitalizations")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(dates, I_pred, label=r"$I_{\mathrm{pred}}$", color=_DARK[1])
    ax.scatter(dates, I_obs, label=r"$I_{\mathrm{obs}}$", color=_LIGHT[1], s=10, alpha=0.4)
    ax.set_title("State Predictions")
    ax.set_ylabel("Population")
    ax.legend()

    ax = axes[1, 0]
    if Rt_true is not None:
        ax.plot(dates, Rt_true, label=r"$R_{t,\mathrm{true}}$", color=_DARK[0])
    ax.plot(
        dates,
        Rt_pred,
        label=r"$R_{t,\mathrm{pred}}$",
        linestyle="--" if Rt_true is not None else "-",
        color=_LIGHT[0] if Rt_true is not None else _DARK[0],
    )
    ax.axhline(y=1, color="red", linestyle=":", alpha=0.5, label="$R_t = 1$")
    ax.set_title("Parameter Recovery")
    ax.set_ylabel("Value")
    top = ax.get_ylim()[1]
    pad = top * 0.10
    ax.set_ylim(-pad, top + pad)
    ax.legend()

    ax = axes[1, 1]
    if sigma_true is not None:
        ax.plot(dates, sigma_true, label=r"$\sigma_{\mathrm{true}}$", color=_DARK[1])
    ax.plot(
        dates,
        sigma_pred,
        label=r"$\sigma_{\mathrm{pred}}$",
        linestyle="--" if sigma_true is not None else "-",
        color=_LIGHT[1] if sigma_true is not None else _DARK[1],
    )
    ax.set_title("Parameter Recovery")
    ax.set_ylabel("Value")
    top = ax.get_ylim()[1]
    pad = top * 0.10
    ax.set_ylim(-pad, top + pad)
    ax.legend()

    fig.autofmt_xdate(rotation=30)
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
