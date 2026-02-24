"""Custom ODE — mathematical definition."""

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
# Keys — define string keys for your state variables and parameters
# ============================================================================

# TODO: define state variable keys
# Y1_KEY = "y1"
# Y2_KEY = "y2"

# TODO: define parameter keys
# PARAM_KEY = "param"

# ============================================================================
# Constants
# ============================================================================

# TODO: set your initial conditions
# Y1_0 = 1.0
# Y2_0 = 0.0

# TODO: set time domain
# T_TOTAL = 10

# ============================================================================
# ODE Definition
# ============================================================================


def my_ode(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    """TODO: implement your ODE system.

    Args:
        x: Independent variable (e.g. time). Shape: (N,)
        y: State variables. Each element has shape (N,)
        args: Parameters (both fixed and learnable).

    Returns:
        dy/dx stacked as a tensor.
    """
    # Example:
    # y1, y2 = y
    # p = args[PARAM_KEY]
    # dy1 = -p(x) * y1
    # dy2 = p(x) * y1 - y2
    # return torch.stack([dy1, dy2])
    raise NotImplementedError("TODO: implement your ODE")


# ============================================================================
# Validation
# ============================================================================

validation: ValidationRegistry = {
    # TODO: add validation sources
    # "param_name": ColumnRef(column="your_column"),
}

# ============================================================================
# Data Module Factory
# ============================================================================


def create_data_module(hp: ODEHyperparameters):
    # TODO: create and return your DataModule
    # See anypinn.catalog for examples of DataModule subclasses.
    raise NotImplementedError("TODO: implement create_data_module")


# ============================================================================
# Problem Factory
# ============================================================================


def create_problem(hp: ODEHyperparameters) -> ODEInverseProblem:
    # TODO: define your problem
    # props = ODEProperties(
    #     ode=my_ode,
    #     y0=torch.tensor([Y1_0, Y2_0]),
    #     args={
    #         # fixed arguments go here
    #     },
    # )
    #
    # fields = FieldsRegistry({
    #     Y1_KEY: Field(config=hp.fields_config),
    #     Y2_KEY: Field(config=hp.fields_config),
    # })
    # params = ParamsRegistry({
    #     PARAM_KEY: Parameter(config=hp.params_config),
    # })
    #
    # def predict_data(
    #     x_data: Tensor, fields: FieldsRegistry, _params: ParamsRegistry
    # ) -> Tensor:
    #     return cast(Tensor, fields[Y1_KEY](x_data))
    #
    # return ODEInverseProblem(
    #     props=props, hp=hp, fields=fields, params=params,
    #     predict_data=predict_data,
    # )
    raise NotImplementedError("TODO: implement create_problem")
