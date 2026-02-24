"""Lotka-Volterra predator-prey — mathematical definition."""

from __future__ import annotations

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

X_KEY = "x"
Y_KEY = "y"
ALPHA_KEY = "alpha"
BETA_KEY = "beta"
DELTA_KEY = "delta"
GAMMA_KEY = "gamma"

# ============================================================================
# Constants
# ============================================================================

# True parameter values
TRUE_ALPHA = 0.5
TRUE_BETA = 0.02
TRUE_DELTA = 0.01
TRUE_GAMMA = 0.5

# Initial conditions (populations)
X0 = 40.0
Y0 = 9.0

# Time domain
T_TOTAL = 50

# Population scaling
POP_SCALE = 100.0

# Noise level for synthetic data (std as fraction of signal)
NOISE_FRAC = 0.02

# ============================================================================
# Fourier encoding
# ============================================================================


def fourier_encode(t: Tensor) -> Tensor:
    features = [t]
    for k in range(1, 7):
        features.append(torch.sin(k * t))
    return torch.cat(features, dim=-1)


# ============================================================================
# ODE Definition
# ============================================================================


def lotka_volterra(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    """Scaled Lotka-Volterra ODE. Populations scaled by POP_SCALE, time by T_TOTAL."""
    prey, predator = y
    b = args[BETA_KEY]
    alpha = args[ALPHA_KEY]
    delta = args[DELTA_KEY]
    gamma = args[GAMMA_KEY]

    dx = alpha(x) * prey - b(x) * prey * predator * POP_SCALE
    dy = delta(x) * prey * predator * POP_SCALE - gamma(x) * predator

    dx = dx * T_TOTAL
    dy = dy * T_TOTAL
    return torch.stack([dx, dy])


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
    from anypinn.catalog.lotka_volterra import LotkaVolterraDataModule

    def LV_unscaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
        prey, predator = y
        b = args[BETA_KEY]
        alpha = args[ALPHA_KEY]
        delta = args[DELTA_KEY]
        gamma = args[GAMMA_KEY]
        dx = alpha(x) * prey - b(x) * prey * predator
        dy = delta(x) * prey * predator - gamma(x) * predator
        return torch.stack([dx, dy])

    gen_props = ODEProperties(
        ode=LV_unscaled,
        y0=torch.tensor([X0, Y0]),
        args={
            ALPHA_KEY: Argument(TRUE_ALPHA),
            BETA_KEY: Argument(TRUE_BETA),
            DELTA_KEY: Argument(TRUE_DELTA),
            GAMMA_KEY: Argument(TRUE_GAMMA),
        },
    )

    return LotkaVolterraDataModule(
        hp=hp,
        gen_props=gen_props,
        noise_frac=NOISE_FRAC,
        validation=validation,
        callbacks=[DataScaling(y_scale=[1 / POP_SCALE, 1 / POP_SCALE])],
    )


# ============================================================================
# Problem Factory
# ============================================================================


def create_problem(hp: ODEHyperparameters) -> ODEInverseProblem:
    props = ODEProperties(
        ode=lotka_volterra,
        y0=torch.tensor([X0, Y0]) / POP_SCALE,
        args={
            ALPHA_KEY: Argument(TRUE_ALPHA),
            DELTA_KEY: Argument(TRUE_DELTA),
            GAMMA_KEY: Argument(TRUE_GAMMA),
        },
    )

    fields = FieldsRegistry(
        {
            X_KEY: Field(config=hp.fields_config),
            Y_KEY: Field(config=hp.fields_config),
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
        x_pred = fields[X_KEY](x_data)
        y_pred = fields[Y_KEY](x_data)
        return torch.stack([x_pred, y_pred])

    return ODEInverseProblem(
        props=props,
        hp=hp,
        fields=fields,
        params=params,
        predict_data=predict_data,
    )
