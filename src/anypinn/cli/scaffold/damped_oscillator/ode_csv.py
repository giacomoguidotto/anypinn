"""Damped oscillator — mathematical definition."""

from __future__ import annotations

import math
from typing import cast

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

X_KEY = "x"
V_KEY = "v"
ZETA_KEY = "zeta"
OMEGA_KEY = "omega0"

# ============================================================================
# Constants
# ============================================================================

# True parameter values
TRUE_ZETA = 0.15
TRUE_OMEGA0 = 2 * math.pi

# Initial conditions
X0 = 1.0
V0 = 0.0

# Time domain (seconds)
T_TOTAL = 5

# Noise level for synthetic data
NOISE_STD = 0.02

# ============================================================================
# ODE Definition
# ============================================================================


def oscillator(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    """Scaled damped oscillator ODE: $dx/dt = v$, $dv/dt = -2 zeta omega_0 v - omega_0^2 x$."""
    pos, vel = y
    z = args[ZETA_KEY]
    omega0 = args[OMEGA_KEY]

    dx = vel
    dv = -2 * z(x) * omega0(x) * vel - omega0(x) ** 2 * pos

    dx = dx * T_TOTAL
    dv = dv * T_TOTAL
    return torch.stack([dx, dv])


# ============================================================================
# Validation
# ============================================================================

validation: ValidationRegistry = {
    # TODO: map zeta to a column in your CSV
    # ZETA_KEY: ColumnRef(column="your_column"),
}

# ============================================================================
# Data Module Factory
# ============================================================================


def create_data_module(hp: ODEHyperparameters):
    from anypinn.catalog.damped_oscillator import DampedOscillatorDataModule

    def oscillator_unscaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
        pos, vel = y
        z = args[ZETA_KEY]
        omega0 = args[OMEGA_KEY]
        dx = vel
        dv = -2 * z(x) * omega0(x) * vel - omega0(x) ** 2 * pos
        return torch.stack([dx, dv])

    gen_props = ODEProperties(
        ode=oscillator_unscaled,
        y0=torch.tensor([X0, V0]),
        args={
            ZETA_KEY: Argument(TRUE_ZETA),
            OMEGA_KEY: Argument(TRUE_OMEGA0),
        },
    )

    return DampedOscillatorDataModule(
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
        ode=oscillator,
        y0=torch.tensor([X0, V0]),
        args={
            OMEGA_KEY: Argument(TRUE_OMEGA0),
        },
    )

    fields = FieldsRegistry(
        {
            X_KEY: Field(config=hp.fields_config),
            V_KEY: Field(config=hp.fields_config),
        }
    )
    params = ParamsRegistry(
        {
            ZETA_KEY: Parameter(config=hp.params_config),
        }
    )

    def predict_data(x_data: Tensor, fields: FieldsRegistry, _params: ParamsRegistry) -> Tensor:
        x_pred = fields[X_KEY](x_data)
        return cast(Tensor, x_pred)

    return ODEInverseProblem(
        props=props,
        hp=hp,
        fields=fields,
        params=params,
        predict_data=predict_data,
    )
