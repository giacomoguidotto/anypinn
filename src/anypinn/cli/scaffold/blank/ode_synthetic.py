"""ODE mathematical definition."""

from __future__ import annotations

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
from anypinn.problems import ODEHyperparameters, ODEInverseProblem, ODEProperties

# TODO: define your ODE system here


validation: ValidationRegistry = {}


def create_data_module(hp: ODEHyperparameters):
    raise NotImplementedError("TODO: implement create_data_module")


def create_problem(hp: ODEHyperparameters) -> ODEInverseProblem:
    raise NotImplementedError("TODO: implement create_problem")
