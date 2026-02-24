"""SEIR epidemic model — mathematical definition."""

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
E_KEY = "E"
I_KEY = "I"
BETA_KEY = "beta"
SIGMA_KEY = "sigma"
GAMMA_KEY = "gamma"

# ============================================================================
# Constants
# ============================================================================

# True parameter values
TRUE_BETA = 0.5
TRUE_SIGMA = 1 / 5.2
TRUE_GAMMA = 1 / 10

# Initial conditions (fractions)
S0 = 0.99
E0 = 0.01
I0 = 0.001

# Time domain
T_DAYS = 160

# Noise level for synthetic data
NOISE_STD = 0.0005

# ============================================================================
# ODE Definition
# ============================================================================


def SEIR(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    """Scaled SEIR ODE system."""
    S, E, I = y
    b = args[BETA_KEY]
    sigma = args[SIGMA_KEY]
    gamma = args[GAMMA_KEY]

    dS = -b(x) * S * I
    dE = b(x) * S * I - sigma(x) * E
    dI = sigma(x) * E - gamma(x) * I

    dS = dS * T_DAYS
    dE = dE * T_DAYS
    dI = dI * T_DAYS
    return torch.stack([dS, dE, dI])


# ============================================================================
# Validation
# ============================================================================

validation: ValidationRegistry = {
    # TODO: map beta to a column in your CSV
    # BETA_KEY: ColumnRef(column="your_column"),
}

# ============================================================================
# Data Module Factory
# ============================================================================


def create_data_module(hp: ODEHyperparameters):
    from anypinn.catalog.seir import SEIRDataModule

    # Unscaled SEIR for data generation
    def SEIR_unscaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
        S, E, I = y
        b = args[BETA_KEY]
        sigma = args[SIGMA_KEY]
        gamma = args[GAMMA_KEY]
        dS = -b(x) * S * I
        dE = b(x) * S * I - sigma(x) * E
        dI = sigma(x) * E - gamma(x) * I
        return torch.stack([dS, dE, dI])

    gen_props = ODEProperties(
        ode=SEIR_unscaled,
        y0=torch.tensor([S0, E0, I0]),
        args={
            BETA_KEY: Argument(TRUE_BETA),
            SIGMA_KEY: Argument(TRUE_SIGMA),
            GAMMA_KEY: Argument(TRUE_GAMMA),
        },
    )

    return SEIRDataModule(
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
        ode=SEIR,
        y0=torch.tensor([S0, E0, I0]),
        args={
            SIGMA_KEY: Argument(TRUE_SIGMA),
            GAMMA_KEY: Argument(TRUE_GAMMA),
        },
    )

    fields = FieldsRegistry(
        {
            S_KEY: Field(config=hp.fields_config),
            E_KEY: Field(config=hp.fields_config),
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
