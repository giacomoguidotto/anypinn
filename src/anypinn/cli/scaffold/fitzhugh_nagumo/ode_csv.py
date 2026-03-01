"""FitzHugh-Nagumo neuron model — mathematical definition."""

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

validation: ValidationRegistry = {
    # TODO: map parameters to columns in your CSV
    # EPSILON_KEY: ColumnRef(column="your_column"),
    # A_KEY: ColumnRef(column="your_column"),
}

# ============================================================================
# Data Module Factory
# ============================================================================


def create_data_module(hp: ODEHyperparameters):
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
        validation=validation,
    )


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
