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

    Rt_pred = preds[Rt_KEY]
    Rt_true = trues[Rt_KEY] if trues else None

    I_pred = C * preds[I_KEY]
    I_data = C * I_data

    # plot
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.lineplot(x=t_data, y=I_pred, label="$I_{pred}$", ax=axes[0])
    sns.lineplot(x=t_data, y=I_data, label="$I_{observed}$", linestyle="--", ax=axes[0])
    axes[0].set_title("Reduced SIR Model Predictions")
    axes[0].set_xlabel("Time (days)")
    axes[0].set_ylabel("I (Population)")
    axes[0].legend()

    sns.lineplot(x=t_data, y=Rt_true, label=r"$R_{t, true}$", ax=axes[1])
    sns.lineplot(x=t_data, y=Rt_pred, label=r"$R_{t, pred}$", linestyle="--", ax=axes[1])

    axes[1].set_title(r"$R_t$ Parameter Prediction")
    axes[1].set_xlabel("Time (days)")
    axes[1].set_ylabel(r"$R_t$")
    axes[1].legend()

    plt.tight_layout()

    fig.savefig(results_dir / f"{experiment_name}.png", dpi=300)

    # save
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
