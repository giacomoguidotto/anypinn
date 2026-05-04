"""SIR epidemic model — mathematical definition."""

from __future__ import annotations

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

S_KEY = "S"
I_KEY = "I"
BETA_KEY = "beta"
DELTA_KEY = "delta"
N_KEY = "N"

# ============================================================================
# Constants
# ============================================================================

# Scaling constants
C = 1e6  # population scale
T = 90  # time scale (days)

# Known parameters
N_POP = 56e6
DELTA = 1 / 5

# True parameter values (for synthetic data / validation)
TRUE_BETA = 0.6

# ============================================================================
# ODE Definition
# ============================================================================


def SIR(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    """Scaled SIR ODE system."""
    S, I = y
    b, d, N = args[BETA_KEY], args[DELTA_KEY], args[N_KEY]

    dS = -b(x) * I * S * C / N(x)
    dI = b(x) * I * S * C / N(x) - d(x) * I

    dS = dS * T
    dI = dI * T
    return torch.stack([dS, dI])


# ============================================================================
# Validation
# ============================================================================

# --- VARIANT: source/synthetic ---
validation_synthetic: ValidationRegistry = {
    BETA_KEY: lambda x: torch.full_like(x, TRUE_BETA),
}
# --- VARIANT: source/csv ---
validation_csv: ValidationRegistry = {
    BETA_KEY: lambda x: torch.full_like(x, TRUE_BETA),
    # To validate against a CSV column instead, use:
    # BETA_KEY: ColumnRef(column="Rt", transform=lambda rt: rt * DELTA),
}
# --- END VARIANT ---

# ============================================================================
# Data Module Factory
# ============================================================================


# --- VARIANT: source/synthetic ---
def create_data_module_synthetic(hp: ODEHyperparameters):
    from anypinn.catalog.sir import SIRInvDataModule

    # Unscaled SIR for data generation (no C or T scaling)
    def SIR_unscaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
        S, I = y
        b, d, N = args[BETA_KEY], args[DELTA_KEY], args[N_KEY]
        dS = -b(x) * S * I / N(x)
        dI = b(x) * S * I / N(x) - d(x) * I
        return torch.stack([dS, dI])

    gen_props = ODEProperties(
        ode=SIR_unscaled,
        y0=torch.tensor([N_POP - 1, 1]),
        args={
            BETA_KEY: Argument(TRUE_BETA),
            DELTA_KEY: Argument(DELTA),
            N_KEY: Argument(N_POP),
        },
    )

    return SIRInvDataModule(
        hp=hp,
        gen_props=gen_props,
        validation=validation_synthetic,
        callbacks=[DataScaling(y_scale=1 / C)],
    )


# --- VARIANT: source/csv ---
def create_data_module_csv(hp: ODEHyperparameters):
    from anypinn.catalog.sir import SIRInvDataModule

    return SIRInvDataModule(
        hp=hp,
        validation=validation_csv,
        callbacks=[DataScaling(y_scale=1 / C)],
    )


# --- END VARIANT ---

# ============================================================================
# Problem Factory
# ============================================================================


def create_problem(hp: ODEHyperparameters) -> ODEInverseProblem:
    props = ODEProperties(
        ode=SIR,
        y0=torch.tensor([N_POP - 1, 1]) / C,
        args={
            DELTA_KEY: Argument(DELTA),
            N_KEY: Argument(N_POP),
        },
    )

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


_DARK = ["#1f77b4", "#ff7f0e", "#2ca02c"]
_LIGHT = ["#aec7e8", "#ffbb78", "#98df8a"]


def plot_and_save(
    predictions: Predictions,
    results_dir: Path,
    experiment_name: str,
) -> None:
    batch, preds, trues = predictions
    t_data, I_data = batch
    I_data = I_data.squeeze(-1)

    t_days = t_data * T

    S_pred = C * preds[S_KEY]
    I_pred = C * preds[I_KEY]
    R_pred = N_POP - S_pred - I_pred

    I_data = C * I_data

    beta_pred = preds[BETA_KEY]
    beta_true = trues[BETA_KEY] if trues else None

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("SIR Model", fontsize=14)

    ax1 = axes[0]
    ax2 = ax1.twinx()

    ax1.plot(t_days, S_pred, label=r"$S_{\mathrm{pred}}$", color=_DARK[0])
    ax1.set_ylabel("$S$ (Population)", color=_DARK[0])
    ax1.tick_params(axis="y", labelcolor=_DARK[0])

    ax2.plot(t_days, I_pred, label=r"$I_{\mathrm{pred}}$", color=_DARK[1])
    ax2.plot(t_days, R_pred, label=r"$R_{\mathrm{pred}}$", color=_DARK[2])
    ax2.scatter(t_days, I_data, label=r"$I_{\mathrm{obs}}$", color=_LIGHT[1], s=10, alpha=0.4)
    ax2.set_ylabel("$I$, $R$ (Population)", color=_DARK[1])
    ax2.tick_params(axis="y", labelcolor=_DARK[1])
    ax2.grid(False)

    ax1.set_title("State Predictions")
    ax1.set_xlabel(r"$t$ (days)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center")
    ax2.legend().remove()

    ax = axes[1]
    if beta_true is not None:
        ax.plot(t_days, beta_true, label=r"$\beta_{\mathrm{true}}$", color=_DARK[0])
    ax.plot(
        t_days,
        beta_pred,
        label=r"$\beta_{\mathrm{pred}}$",
        linestyle="--" if beta_true is not None else "-",
        color=_LIGHT[0] if beta_true is not None else _DARK[0],
    )
    ax.set_title("Parameter Recovery")
    ax.set_xlabel(r"$t$ (days)")
    ax.set_ylabel("Value")
    ax.legend()

    plt.tight_layout()
    fig.savefig(results_dir / "plot.png", dpi=300)
    plt.close(fig)

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
    df.to_csv(results_dir / "data.csv", index=False, float_format="%.6e")
