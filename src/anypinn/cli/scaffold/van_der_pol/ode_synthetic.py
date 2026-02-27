"""Van der Pol oscillator — mathematical definition."""

from __future__ import annotations

import torch
from torch import Tensor

from anypinn.core import (
    ArgsRegistry,
    Argument,
    Field,
    FieldsRegistry,
    Parameter,
    ParamsRegistry,
    ValidationRegistry,
)
from anypinn.lightning.callbacks import DataScaling
from anypinn.problems import ODEHyperparameters, ODEInverseProblem, ODEProperties

# ============================================================================
# Keys
# ============================================================================

U_KEY = "u"
MU_KEY = "mu"

# ============================================================================
# Constants
# ============================================================================

# True parameter value
TRUE_MU = 1.0

# Initial conditions
U0 = 2.0
DU0 = 0.0

# Time domain (seconds)
T_TOTAL = 20

# Noise level for synthetic data
NOISE_STD = 0.05

# ============================================================================
# ODE Definition
# ============================================================================


def vdp_scaled(
    x: Tensor,
    y: Tensor,
    args: ArgsRegistry,
    derivs: list[Tensor] | None = None,
) -> Tensor:
    """Native second-order Van der Pol ODE (scaled time)."""
    assert derivs is not None
    u = y[0]  # (m, 1)
    du_dtau = derivs[0][0]  # (m, 1) — derivative w.r.t. scaled time tau in [0,1]
    mu = args[MU_KEY]
    # Physical ODE: d2u/dt2 = mu*(1-u^2)*du/dt - u
    # With tau = t/T: d2u/dtau2 = T*mu*(1-u^2)*du/dtau - T^2*u
    return (T_TOTAL * mu(x) * (1 - u**2) * du_dtau - T_TOTAL**2 * u).unsqueeze(0)


# ============================================================================
# Validation
# ============================================================================

validation: ValidationRegistry = {
    MU_KEY: lambda x: torch.full_like(x, TRUE_MU),
}

# ============================================================================
# Data Module Factory
# ============================================================================


def create_data_module(hp: ODEHyperparameters):
    from anypinn.catalog.van_der_pol import VanDerPolDataModule

    def vdp_unscaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
        u, v = y
        mu = args[MU_KEY]
        du = v
        dv = mu(x) * (1 - u**2) * v - u
        return torch.stack([du, dv])

    gen_props = ODEProperties(
        ode=vdp_unscaled,
        y0=torch.tensor([U0, DU0]),
        args={
            MU_KEY: Argument(TRUE_MU),
        },
    )

    return VanDerPolDataModule(
        hp=hp,
        gen_props=gen_props,
        noise_std=NOISE_STD,
        validation=validation,
        callbacks=[DataScaling(y_scale=1.0)],
    )


# ============================================================================
# Problem Factory
# ============================================================================


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
