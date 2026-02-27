"""Lorenz system — mathematical definition."""

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

X_KEY = "x"
Y_KEY = "y"
Z_KEY = "z"
SIGMA_KEY = "sigma"
RHO_KEY = "rho"
BETA_KEY = "beta"

# ============================================================================
# Constants
# ============================================================================

# True parameter values
TRUE_SIGMA = 10.0
TRUE_RHO = 28.0
TRUE_BETA = 8.0 / 3.0

# Initial conditions
X0 = -8.0
Y0 = 7.0
Z0 = 27.0

# Time domain
T_TOTAL = 3

# State scaling (Lorenz variables range ~±20-45)
SCALE = 20.0

# Noise level for synthetic data (additive Gaussian)
NOISE_STD = 0.5

# ============================================================================
# ODE Definition
# ============================================================================


def lorenz_scaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    """Scaled Lorenz ODE. States pre-divided by SCALE, time by T_TOTAL."""
    lx, ly, lz = y
    sigma = args[SIGMA_KEY]
    rho = args[RHO_KEY]
    beta = args[BETA_KEY]

    dx = sigma(x) * (ly - lx)
    dy = lx * (rho(x) / SCALE - lz) - ly
    dz = lx * ly * SCALE - beta(x) * lz

    dx = dx * T_TOTAL
    dy = dy * T_TOTAL
    dz = dz * T_TOTAL
    return torch.stack([dx, dy, dz])


# ============================================================================
# Validation
# ============================================================================

validation: ValidationRegistry = {
    # TODO: map parameters to columns in your CSV
    # SIGMA_KEY: ColumnRef(column="your_column"),
    # RHO_KEY: ColumnRef(column="your_column"),
    # BETA_KEY: ColumnRef(column="your_column"),
}

# ============================================================================
# Data Module Factory
# ============================================================================


def create_data_module(hp: ODEHyperparameters):
    from anypinn.catalog.lorenz import LorenzDataModule

    def lorenz_unscaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
        lx, ly, lz = y
        sigma = args[SIGMA_KEY]
        rho = args[RHO_KEY]
        beta = args[BETA_KEY]
        dx = sigma(x) * (ly - lx)
        dy = lx * (rho(x) - lz) - ly
        dz = lx * ly - beta(x) * lz
        return torch.stack([dx, dy, dz])

    gen_props = ODEProperties(
        ode=lorenz_unscaled,
        y0=torch.tensor([X0, Y0, Z0]),
        args={
            SIGMA_KEY: Argument(TRUE_SIGMA),
            RHO_KEY: Argument(TRUE_RHO),
            BETA_KEY: Argument(TRUE_BETA),
        },
    )

    return LorenzDataModule(
        hp=hp,
        gen_props=gen_props,
        noise_std=NOISE_STD,
        validation=validation,
        callbacks=[DataScaling(y_scale=[1 / SCALE, 1 / SCALE, 1 / SCALE])],
    )


# ============================================================================
# Problem Factory
# ============================================================================


def create_problem(hp: ODEHyperparameters) -> ODEInverseProblem:
    props = ODEProperties(
        ode=lorenz_scaled,
        y0=torch.tensor([X0, Y0, Z0]) / SCALE,
        args={},
    )

    fields = FieldsRegistry(
        {
            X_KEY: Field(config=hp.fields_config),
            Y_KEY: Field(config=hp.fields_config),
            Z_KEY: Field(config=hp.fields_config),
        }
    )
    params = ParamsRegistry(
        {
            SIGMA_KEY: Parameter(config=hp.params_config),
            RHO_KEY: Parameter(config=hp.params_config),
            BETA_KEY: Parameter(config=hp.params_config),
        }
    )

    def predict_data(x_data: Tensor, fields: FieldsRegistry, _params: ParamsRegistry) -> Tensor:
        x_pred = fields[X_KEY](x_data)
        y_pred = fields[Y_KEY](x_data)
        z_pred = fields[Z_KEY](x_data)
        return torch.stack([x_pred, y_pred, z_pred], dim=1)

    return ODEInverseProblem(
        props=props,
        hp=hp,
        fields=fields,
        params=params,
        predict_data=predict_data,
    )
