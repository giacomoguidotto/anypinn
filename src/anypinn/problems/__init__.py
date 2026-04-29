"""Problem templates and implementations.

This module provides ready-made constraint types for ODE and PDE problems,
plus the ``ODEInverseProblem`` convenience class that wires them together.

## ODE vs PDE constraints

ODE and PDE problems share the same ``Constraint`` base class from
``anypinn.core``, but differ in the physics they enforce:

**ODE constraints** (``anypinn.problems.ode``):

- ``ResidualsConstraint``: minimizes the ODE residual
  ||dy/dt - f(t, y, args)||. Supports arbitrary-order ODEs via the
  ``order`` parameter on ``ODEProperties``.
- ``ICConstraint``: enforces initial conditions y(t0) = Y0 (and derivative
  ICs for higher-order ODEs).
- ``DataConstraint``: enforces fit to observed data ||y_hat - y_obs||.

**PDE constraints** (``anypinn.problems.pde``):

- ``PDEResidualConstraint``: minimizes a user-defined PDE residual function
  over interior collocation points.
- ``DirichletBCConstraint``: enforces u(x_bc) = g(x_bc).
- ``NeumannBCConstraint``: enforces du/dn(x_bc) = h(x_bc).
- ``PeriodicBCConstraint``: enforces u(x_left) = u(x_right) and matching
  derivatives.

## Forward vs inverse direction

The difference between forward and inverse problems is which values are
known and which are learned:

- **Forward**: the PDE/ODE parameters (e.g. diffusivity, reaction rates) are
  known constants passed as ``Argument`` values. The neural field learns the
  solution. Constraints: residual + boundary/initial conditions only.
- **Inverse**: some parameters are unknown and passed as ``Parameter``
  instances (learnable). A ``DataConstraint`` is added so observed data
  guides the optimizer toward the correct parameter values alongside the
  solution.

Switching direction is a matter of moving entries between ``args`` and
``params`` — see ``anypinn.core`` for the Argument/Parameter promotion
pattern.

## Writing a custom Constraint

Subclass ``anypinn.core.Constraint`` and implement the ``loss`` method:

```python
class MyConstraint(Constraint):
    def loss(self, batch, criterion, log=None):
        (x_data, y_data), x_coll = batch
        # compute your physics loss here
        loss = ...
        if log is not None:
            log("loss/my_term", loss)
        return loss
```

Optionally override ``inject_context(context)`` if the constraint needs
domain bounds or validation data at runtime.
"""

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
    PDEResidualConstraint,
    PDEResidualFn,
    PeriodicBCConstraint,
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
    "PDEResidualConstraint",
    "PDEResidualFn",
    "PeriodicBCConstraint",
    "PredictDataFn",
    "ResidualsConstraint",
]
