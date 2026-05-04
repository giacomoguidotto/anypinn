"""FitzHugh-Nagumo neuron model — mathematical definition."""

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
    Field,
    FieldsRegistry,
    Parameter,
    ParamsRegistry,
    Predictions,
    ValidationRegistry,
)
from anypinn.problems import ODEHyperparameters, ODEInverseProblem, ODEProperties

# ============================================================================
# Keys
# ============================================================================

V_KEY = "v"
W_KEY = "w"
EPSILON_KEY = "epsilon"
A_KEY = "a"

# ============================================================================
# Constants
# ============================================================================

# True parameter values (to recover)
TRUE_EPSILON = 0.08
TRUE_A = 0.7

# Fixed constants (not trainable)
B = 0.8
I_EXT = 0.5

# Initial conditions
V0 = -1.0
W0 = 1.0

# Time domain
T_TOTAL = 50

# Noise level for synthetic data (additive Gaussian)
NOISE_STD = 0.05

# ============================================================================
# ODE Definition
# ============================================================================


def fhn_scaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    """Scaled FHN ODE for training. Time scaled by T_TOTAL."""
    v, w = y
    eps = args[EPSILON_KEY]
    a = args[A_KEY]

    dv = (v - v**3 / 3 - w + I_EXT) * T_TOTAL
    dw = eps(x) * (v + a(x) - B * w) * T_TOTAL
    return torch.stack([dv, dw])


# ============================================================================
# Validation
# ============================================================================

# --- VARIANT: source/synthetic ---
validation_synthetic: ValidationRegistry = {
    EPSILON_KEY: lambda x: torch.full_like(x, TRUE_EPSILON),
    A_KEY: lambda x: torch.full_like(x, TRUE_A),
}
# --- VARIANT: source/csv ---
validation_csv: ValidationRegistry = {
    # TODO: map parameters to columns in your CSV
    # EPSILON_KEY: ColumnRef(column="your_column"),
    # A_KEY: ColumnRef(column="your_column"),
}
# --- END VARIANT ---

# ============================================================================
# Data Module Factory
# ============================================================================


# --- VARIANT: source/synthetic ---
def create_data_module_synthetic(hp: ODEHyperparameters):
    from anypinn.catalog.fitzhugh_nagumo import FitzHughNagumoDataModule

    def fhn_unscaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
        v, w = y
        eps = args[EPSILON_KEY]
        a = args[A_KEY]
        dv = v - v**3 / 3 - w + I_EXT
        dw = eps(x) * (v + a(x) - B * w)
        return torch.stack([dv, dw])

    gen_props = ODEProperties(
        ode=fhn_unscaled,
        y0=torch.tensor([V0, W0]),
        args={
            EPSILON_KEY: Argument(TRUE_EPSILON),
            A_KEY: Argument(TRUE_A),
        },
    )

    return FitzHughNagumoDataModule(
        hp=hp,
        gen_props=gen_props,
        noise_std=NOISE_STD,
        validation=validation_synthetic,
    )


# --- VARIANT: source/csv ---
def create_data_module_csv(hp: ODEHyperparameters):
    from anypinn.catalog.fitzhugh_nagumo import FitzHughNagumoDataModule

    def fhn_unscaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
        v, w = y
        eps = args[EPSILON_KEY]
        a = args[A_KEY]
        dv = v - v**3 / 3 - w + I_EXT
        dw = eps(x) * (v + a(x) - B * w)
        return torch.stack([dv, dw])

    gen_props = ODEProperties(
        ode=fhn_unscaled,
        y0=torch.tensor([V0, W0]),
        args={
            EPSILON_KEY: Argument(TRUE_EPSILON),
            A_KEY: Argument(TRUE_A),
        },
    )

    return FitzHughNagumoDataModule(
        hp=hp,
        gen_props=gen_props,
        noise_std=NOISE_STD,
        validation=validation_csv,
    )


# --- END VARIANT ---

# ============================================================================
# Problem Factory
# ============================================================================


def create_problem(hp: ODEHyperparameters) -> ODEInverseProblem:
    props = ODEProperties(
        ode=fhn_scaled,
        y0=torch.tensor([V0, W0]),
        args={},
    )

    fields = FieldsRegistry(
        {
            V_KEY: Field(config=hp.fields_config),
            W_KEY: Field(config=hp.fields_config),
        }
    )
    params = ParamsRegistry(
        {
            EPSILON_KEY: Parameter(config=hp.params_config),
            A_KEY: Parameter(config=hp.params_config),
        }
    )

    def predict_data(x_data: Tensor, fields: FieldsRegistry, _params: ParamsRegistry) -> Tensor:
        v_pred = fields[V_KEY](x_data)
        return v_pred.unsqueeze(1)  # (N, 1, 1) — only v is observed

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
    t_data, y_data = batch

    v_pred = preds[V_KEY]
    w_pred = preds[W_KEY]
    v_obs = y_data[:, 0].squeeze(-1)

    eps_pred = preds[EPSILON_KEY]
    a_pred = preds[A_KEY]
    eps_true = trues[EPSILON_KEY] if trues else None
    a_true = trues[A_KEY] if trues else None

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(r"FitzHugh--Nagumo Model", fontsize=14)

    ax = axes[0]
    sns.lineplot(x=t_data, y=v_pred, label=r"$v_{\mathrm{pred}}$", ax=ax, color=_DARK[0])
    sns.scatterplot(
        x=t_data, y=v_obs, label=r"$v_{\mathrm{obs}}$", ax=ax, color=_LIGHT[0], s=10, alpha=0.4
    )
    ax.set_title("State Predictions")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("$v$")
    ax.legend()

    ax = axes[1]
    sns.lineplot(x=t_data, y=w_pred, label=r"$w_{\mathrm{pred}}$", ax=ax, color=_DARK[1])
    ax.set_title("Latent State")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("$w$")
    ax.legend()

    ax = axes[2]
    if eps_true is not None:
        sns.lineplot(
            x=t_data, y=eps_true, label=r"$\varepsilon_{\mathrm{true}}$", ax=ax, color=_DARK[0]
        )
    sns.lineplot(
        x=t_data,
        y=eps_pred,
        label=r"$\varepsilon_{\mathrm{pred}}$",
        linestyle="--" if eps_true is not None else "-",
        ax=ax,
        color=_LIGHT[0] if eps_true is not None else _DARK[0],
    )
    if a_true is not None:
        sns.lineplot(x=t_data, y=a_true, label=r"$a_{\mathrm{true}}$", ax=ax, color=_DARK[1])
    sns.lineplot(
        x=t_data,
        y=a_pred,
        label=r"$a_{\mathrm{pred}}$",
        linestyle="--" if a_true is not None else "-",
        ax=ax,
        color=_LIGHT[1] if a_true is not None else _DARK[1],
    )
    ax.set_title("Parameter Recovery")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Value")
    top = ax.get_ylim()[1]
    pad = top * 0.10
    ax.set_ylim(-pad, top + pad)
    ax.legend()

    plt.tight_layout()
    fig.savefig(results_dir / "plot.png", dpi=300)
    plt.close(fig)

    df = pd.DataFrame(
        {
            "t": t_data,
            "v_obs": v_obs,
            "v_pred": v_pred,
            "w_pred": w_pred,
            "epsilon_pred": eps_pred,
            "a_pred": a_pred,
            "epsilon_true": eps_true,
            "a_true": a_true,
        }
    )
    df.to_csv(results_dir / "data.csv", index=False, float_format="%.6e")
