"""Problem templates and implementations."""

from anypinn.problems.ode import (
    DataConstraint,
    ICConstraint,
    ODECallable,
    ODEHyperparameters,
    ODEInverseProblem,
    ODEProperties,
    PredictDataFn,
    ResidualsConstraint,
)
from anypinn.problems.pde import (
    BCValueFn,
    BoundaryCondition,
    DirichletBCConstraint,
    NeumannBCConstraint,
)

__all__ = [
    "BCValueFn",
    "BoundaryCondition",
    "DataConstraint",
    "DirichletBCConstraint",
    "ICConstraint",
    "NeumannBCConstraint",
    "ODECallable",
    "ODEHyperparameters",
    "ODEInverseProblem",
    "ODEProperties",
    "PredictDataFn",
    "ResidualsConstraint",
]
