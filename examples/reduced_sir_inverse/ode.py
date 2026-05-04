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

C = 1e6
T = 120
START_DATE = "2020-02-01"
N_DAYS = 118


# ============================================================================
# ODE Function
# ============================================================================


def rSIR_s(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    I = y
    d, Rt = args[DELTA_KEY], args[Rt_KEY]

    dI = d(x) * (Rt(x) - 1) * I
    dI = dI * T
    return dI


# ============================================================================
# Problem Definition (module-level props for use in plot_and_save)
# ============================================================================

props = ODEProperties(
    ode=rSIR_s,
    y0=torch.tensor([1]) / C,
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
    }
    return SIRInvDataModule(
        hp=hp,
        validation=validation,
        callbacks=[DataScaling(y_scale=1 / C)],
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


_DARK = ["#1f77b4", "#ff7f0e"]
_LIGHT = ["#aec7e8", "#ffbb78"]


def plot_and_save(
    predictions: Predictions,
    results_dir: Path,
    experiment_name: str,
    props: ODEProperties,
    C: float,
) -> None:
    batch, preds, trues = predictions
    t_data, I_data = batch
    I_data = I_data.squeeze(-1)

    dates = pd.to_datetime(START_DATE) + pd.to_timedelta(t_data * N_DAYS, unit="D")

    Rt_pred = preds[Rt_KEY]
    Rt_true = trues[Rt_KEY] if trues else None

    I_pred = C * preds[I_KEY]
    I_data = C * I_data

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Reduced SIR Model", fontsize=14)

    ax = axes[0]
    ax.plot(dates, I_pred, label=r"$I_{\mathrm{pred}}$", color=_DARK[0])
    ax.scatter(dates, I_data, label=r"$I_{\mathrm{obs}}$", color=_LIGHT[0], s=10, alpha=0.4)
    ax.set_title("State Predictions")
    ax.set_ylabel("Population")
    ax.legend()

    ax = axes[1]
    if Rt_true is not None:
        ax.plot(dates, Rt_true, label=r"$R_{t,\mathrm{true}}$", color=_DARK[0])
    ax.plot(
        dates,
        Rt_pred,
        label=r"$R_{t,\mathrm{pred}}$",
        linestyle="--" if Rt_true is not None else "-",
        color=_LIGHT[0] if Rt_true is not None else _DARK[0],
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

    df = pd.DataFrame(
        {
            "t": t_data,
            "I_observed": I_data,
            "I_pred": I_pred,
            "Rt_pred": Rt_pred,
            "Rt_true": Rt_true,
        }
    )
    df.to_csv(results_dir / f"{experiment_name}.csv", index=False, float_format="%.6e")
