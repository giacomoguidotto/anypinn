from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import Tensor

from anypinn.catalog.sir import BETA_KEY, DELTA_KEY, I_KEY, N_KEY, S_KEY, SIRInvDataModule
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
T = 90
N_POP = 56e6
DELTA = 1 / 5


# ============================================================================
# ODE Function
# ============================================================================


def SIR_s(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    S, I = y
    b, d, N = args[BETA_KEY], args[DELTA_KEY], args[N_KEY]

    dS = -b(x) * I * S * C / N(x)
    dI = b(x) * I * S * C / N(x) - d(x) * I

    dS = dS * T
    dI = dI * T
    return torch.stack([dS, dI])


# ============================================================================
# Problem Definition (module-level props for use in plot_and_save)
# ============================================================================

props = ODEProperties(
    ode=SIR_s,
    y0=torch.tensor([N_POP - 1, 1]) / C,
    args={
        DELTA_KEY: Argument(DELTA),
        N_KEY: Argument(N_POP),
    },
)


# ============================================================================
# Data and Problem Factories
# ============================================================================


def create_data_module(hp: ODEHyperparameters) -> SIRInvDataModule:
    validation: ValidationRegistry = {
        BETA_KEY: ColumnRef(column="Rt", transform=lambda rt: rt * DELTA),
    }
    return SIRInvDataModule(
        hp=hp,
        validation=validation,
        callbacks=[DataScaling(y_scale=1 / C)],
    )


def create_problem(hp: ODEHyperparameters) -> ODEInverseProblem:
    fields = FieldsRegistry(
        {
            S_KEY: Field(config=hp.fields_config),
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
    props: ODEProperties,
    C: float,
) -> None:
    batch, preds, trues = predictions
    t_data, I_data = batch
    I_data = I_data.squeeze(-1)

    N = props.args[N_KEY](t_data)

    S_pred = C * preds[S_KEY]
    I_pred = C * preds[I_KEY]
    R_pred = N - S_pred - I_pred

    I_data = C * I_data

    beta_pred = preds[BETA_KEY]
    beta_true = trues[BETA_KEY] if trues else None

    # plot
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    ax2 = ax1.twinx()

    sns.lineplot(x=t_data, y=S_pred, label="$S_{pred}$", ax=ax1, color="C0")
    ax1.set_ylabel("S (Population)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")

    sns.lineplot(x=t_data, y=I_pred, label="$I_{pred}$", ax=ax2, color="C3")
    sns.lineplot(x=t_data, y=R_pred, label="$R_{pred}$", ax=ax2, color="C3")
    sns.lineplot(x=t_data, y=I_data, label="$I_{observed}$", linestyle="--", ax=ax2, color="C1")
    ax2.set_ylabel("I, R (Population)", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax2.grid(False)  # disable grid on secondary axis to avoid overlap with legend

    ax1.set_title("SIR Model Predictions")
    ax1.set_xlabel("Time (days)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center")
    ax2.legend().remove()

    if beta_true is not None:
        sns.lineplot(x=t_data, y=beta_true, label=r"$\beta_{true}$", ax=axes[1])
    sns.lineplot(x=t_data, y=beta_pred, label=r"$\beta_{pred}$", linestyle="--", ax=axes[1])

    axes[1].set_title(r"$\beta$ Parameter Prediction")
    axes[1].set_xlabel("Time (days)")
    axes[1].set_ylabel(r"$\beta$")
    axes[1].legend()

    plt.tight_layout()

    fig.savefig(results_dir / f"{experiment_name}.png", dpi=300)

    # save
    df = pd.DataFrame(
        {
            "t": t_data,
            "I_observed": I_data,
            "S_pred": S_pred,
            "I_pred": I_pred,
            "R_pred": R_pred,
            "beta_pred": beta_pred,
            "beta_true": beta_true,
        }
    )

    df.to_csv(results_dir / f"{experiment_name}.csv", index=False, float_format="%.6e")
