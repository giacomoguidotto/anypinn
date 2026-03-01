from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import Tensor

from anypinn.catalog.fitzhugh_nagumo import (
    A_KEY,
    EPSILON_KEY,
    V_KEY,
    W_KEY,
    FitzHughNagumoDataModule,
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
from anypinn.problems import ODEHyperparameters, ODEInverseProblem, ODEProperties

# ============================================================================
# Constants
# ============================================================================

TRUE_EPSILON = 0.08
TRUE_A = 0.7
B = 0.8
I_EXT = 0.5

V0 = -1.0
W0 = 1.0

T_TOTAL = 50
NOISE_STD = 0.05


# ============================================================================
# ODE Functions
# ============================================================================


def fhn_unscaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    """FitzHugh-Nagumo system for odeint data generation (physical units)."""
    v, w = y
    eps = args[EPSILON_KEY]
    a = args[A_KEY]

    dv = v - v**3 / 3 - w + I_EXT
    dw = eps(x) * (v + a(x) - B * w)
    return torch.stack([dv, dw])


def fhn_scaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    """Scaled FHN ODE for training. Time scaled by T_TOTAL."""
    v, w = y
    eps = args[EPSILON_KEY]
    a = args[A_KEY]

    dv = (v - v**3 / 3 - w + I_EXT) * T_TOTAL
    dw = eps(x) * (v + a(x) - B * w) * T_TOTAL
    return torch.stack([dv, dw])


# ============================================================================
# Data and Problem Factories
# ============================================================================

validation: ValidationRegistry = {
    EPSILON_KEY: lambda x: torch.full_like(x, TRUE_EPSILON),
    A_KEY: lambda x: torch.full_like(x, TRUE_A),
}

gen_props = ODEProperties(
    ode=fhn_unscaled,
    y0=torch.tensor([V0, W0]),
    args={
        EPSILON_KEY: Argument(TRUE_EPSILON),
        A_KEY: Argument(TRUE_A),
    },
)


def create_data_module(hp: ODEHyperparameters) -> FitzHughNagumoDataModule:
    return FitzHughNagumoDataModule(
        hp=hp,
        gen_props=gen_props,
        noise_std=NOISE_STD,
        validation=validation,
    )


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
        return v_pred.unsqueeze(1)  # (N, 1, 1)

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

    v_pred = preds[V_KEY]
    w_pred = preds[W_KEY]
    v_obs = y_data[:, 0].squeeze(-1)

    eps_pred = preds[EPSILON_KEY]
    a_pred = preds[A_KEY]
    eps_true = trues[EPSILON_KEY] if trues else None
    a_true = trues[A_KEY] if trues else None

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1: v(t) predicted vs observed
    ax = axes[0]
    sns.lineplot(x=t_data, y=v_pred, label="$v_{pred}$", ax=ax, color="C0")
    sns.scatterplot(x=t_data, y=v_obs, label="$v_{obs}$", ax=ax, color="C0", s=10, alpha=0.3)
    ax.set_title("Membrane Potential $v(t)$")
    ax.set_xlabel("Time")
    ax.set_ylabel("$v$")
    ax.legend()

    # Panel 2: w(t) predicted only (unobserved)
    ax = axes[1]
    sns.lineplot(x=t_data, y=w_pred, label="$w_{pred}$", ax=ax, color="C1")
    ax.set_title("Recovery Variable $w(t)$ (unobserved)")
    ax.set_xlabel("Time")
    ax.set_ylabel("$w$")
    ax.legend()

    # Panel 3: epsilon and a predictions
    ax = axes[2]
    if eps_true is not None:
        sns.lineplot(x=t_data, y=eps_true, label=r"$\varepsilon_{true}$", ax=ax, color="C0")
    sns.lineplot(
        x=t_data,
        y=eps_pred,
        label=r"$\varepsilon_{pred}$",
        linestyle="--" if eps_true is not None else "-",
        ax=ax,
        color="C3" if eps_true is not None else "C0",
    )
    if a_true is not None:
        sns.lineplot(x=t_data, y=a_true, label="$a_{true}$", ax=ax, color="C1")
    sns.lineplot(
        x=t_data,
        y=a_pred,
        label="$a_{pred}$",
        linestyle="--" if a_true is not None else "-",
        ax=ax,
        color="C4" if a_true is not None else "C1",
    )
    ax.set_title(r"$\varepsilon$, $a$ Predictions")
    ax.set_xlabel("Time")
    ax.set_ylabel("Parameter value")
    ax.legend()

    plt.tight_layout()
    fig.savefig(results_dir / f"{experiment_name}.png", dpi=300)
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
    df.to_csv(results_dir / f"{experiment_name}.csv", index=False, float_format="%.6e")
