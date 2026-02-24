"""ODE mathematical definition."""

from __future__ import annotations

from anypinn.core import ValidationRegistry
from anypinn.problems import ODEHyperparameters, ODEInverseProblem

# TODO: define your ODE system here


validation: ValidationRegistry = {}


def create_data_module(hp: ODEHyperparameters):
    raise NotImplementedError("TODO: implement create_data_module")


def create_problem(hp: ODEHyperparameters) -> ODEInverseProblem:
    raise NotImplementedError("TODO: implement create_problem")
