"""SIR epidemic model — mathematical definition."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor

from anypinn.core import (
    ArgsRegistry,
    Argument,
    ColumnRef,
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
T = 90   # time scale (days)

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

validation: ValidationRegistry = {
    BETA_KEY: lambda x: torch.full_like(x, TRUE_BETA),
}

# ============================================================================
# Data Module Factory
# ============================================================================


def create_data_module(hp: ODEHyperparameters):
    from anypinn.catalog.sir import SIRInvDataModule

    return SIRInvDataModule(
        hp=hp,
        validation=validation,
        callbacks=[DataScaling(y_scale=1 / C)],
    )


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

    def predict_data(
        x_data: Tensor, fields: FieldsRegistry, _params: ParamsRegistry
    ) -> Tensor:
        I_pred = fields[I_KEY](x_data)
        return cast(Tensor, I_pred)

    return ODEInverseProblem(
        props=props,
        hp=hp,
        fields=fields,
        params=params,
        predict_data=predict_data,
    )
