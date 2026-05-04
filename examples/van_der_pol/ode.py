from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import Tensor

from anypinn.catalog.van_der_pol import MU_KEY, U_KEY, VanDerPolDataModule
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

# True parameter value
TRUE_MU = 1.0

# Initial conditions
U0 = 2.0
DU0 = 0.0

# Time domain
T_TOTAL = 20
T = T_TOTAL

# Noise level for synthetic data
NOISE_STD = 0.05


# ============================================================================
# ODE Functions
# ============================================================================


def vdp_unscaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    """First-order 2-state system for odeint data generation: [u, v]."""
    u, v = y
    mu = args[MU_KEY]
    du = v
    dv = mu(x) * (1 - u**2) * v - u
    return torch.stack([du, dv])


def vdp_scaled(
    x: Tensor,
    y: Tensor,
    args: ArgsRegistry,
    derivs: list[Tensor] | None = None,
) -> Tensor:
    """Native second-order ODE for training. Receives derivs[0] = du/d(tau)."""
    assert derivs is not None
    u = y[0]  # (m, 1)
    du_dtau = derivs[0][0]  # (m, 1) — derivative w.r.t. scaled time tau in [0,1]
    mu = args[MU_KEY]
    # Physical ODE: d2u/dt2 = mu*(1-u^2)*du/dt - u
    # With tau = t/T: d2u/dtau2 = T*mu*(1-u^2)*du/dtau - T^2*u
    return (T * mu(x) * (1 - u**2) * du_dtau - T**2 * u).unsqueeze(0)


# ============================================================================
# Data and Problem Factories
# ============================================================================

validation: ValidationRegistry = {
    MU_KEY: lambda x: torch.full_like(x, TRUE_MU),
}

gen_props = ODEProperties(
    ode=vdp_unscaled,
    y0=torch.tensor([U0, DU0]),
    args={
        MU_KEY: Argument(TRUE_MU),
    },
)


def create_data_module(hp: ODEHyperparameters) -> VanDerPolDataModule:
    return VanDerPolDataModule(
        hp=hp,
        gen_props=gen_props,
        noise_std=NOISE_STD,
        validation=validation,
        callbacks=[DataScaling(y_scale=1.0)],
    )


def create_problem(hp: ODEHyperparameters) -> ODEInverseProblem:
    props = ODEProperties(
        ode=vdp_scaled,
        y0=torch.tensor([U0]),
        order=2,
        dy0=[torch.tensor([DU0])],
        args={},
    )

    fields = FieldsRegistry(
        {
            U_KEY: Field(config=hp.fields_config),
        }
    )
    params = ParamsRegistry(
        {
            MU_KEY: Parameter(config=hp.params_config),
        }
    )

    def predict_data(x_data: Tensor, fields: FieldsRegistry, _params: ParamsRegistry) -> Tensor:
        u_pred = fields[U_KEY](x_data)
        return u_pred.unsqueeze(1)

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
) -> None:
    batch, preds, trues = predictions
    t_data, u_data = batch
    u_data = u_data.squeeze(-1)

    u_pred = preds[U_KEY]

    mu_pred = preds[MU_KEY]
    mu_true = trues[MU_KEY] if trues else None

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Van der Pol Oscillator", fontsize=14)

    ax = axes[0]
    sns.lineplot(x=t_data, y=u_pred, label=r"$u_{\mathrm{pred}}$", ax=ax, color=_DARK[0])
    sns.scatterplot(
        x=t_data, y=u_data, label=r"$u_{\mathrm{obs}}$", ax=ax, color=_LIGHT[0], s=10, alpha=0.4
    )
    ax.set_title("State Predictions")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Amplitude")
    ax.legend()

    ax = axes[1]
    if mu_true is not None:
        sns.lineplot(x=t_data, y=mu_true, label=r"$\mu_{\mathrm{true}}$", ax=ax, color=_DARK[0])
    sns.lineplot(
        x=t_data,
        y=mu_pred,
        label=r"$\mu_{\mathrm{pred}}$",
        linestyle="--" if mu_true is not None else "-",
        ax=ax,
        color=_LIGHT[0] if mu_true is not None else _DARK[0],
    )
    ax.set_title("Parameter Recovery")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("Value")
    top = ax.get_ylim()[1]
    pad = top * 0.10
    ax.set_ylim(-pad, top + pad)
    ax.legend()

    plt.tight_layout()
    fig.savefig(results_dir / f"{experiment_name}.png", dpi=300)
    plt.close(fig)

    # save
    df = pd.DataFrame(
        {
            "t": t_data,
            "u_observed": u_data,
            "u_pred": u_pred,
            "mu_pred": mu_pred,
            "mu_true": mu_true,
        }
    )

    df.to_csv(results_dir / f"{experiment_name}.csv", index=False, float_format="%.6e")
