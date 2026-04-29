"""Core PINN building blocks.

This module provides the foundational abstractions for defining and solving
physics-informed neural network problems.

## Argument and Parameter

An ``Argument`` wraps a fixed value (float or callable) that an ODE/PDE
function receives. A ``Parameter`` is a **learnable** ``Argument`` — it
inherits from both ``nn.Module`` and ``Argument``, so it participates in
gradient computation while exposing the same call interface.

To promote a fixed constant to a learnable parameter, replace:

```python
# Fixed: beta = 0.3 throughout training
args = {"beta": Argument(0.3)}
params = {}
```

with:

```python
# Learnable: beta starts at 0.3, the optimizer adjusts it
args = {}
params = {"beta": Parameter(ScalarConfig(init_value=0.3))}
```

The ODE/PDE function signature stays the same either way:

```python
def my_ode(x, y, args):
    beta = args["beta"](x)  # works for both Argument and Parameter
    ...
```

This works because ``ResidualsConstraint`` merges ``params`` into ``args``
before calling the ODE function, and ``Parameter`` is a subclass of
``Argument``. For function-valued parameters (e.g. beta(t) that varies over
the domain), use ``Parameter(MLPConfig(...))`` instead of ``ScalarConfig``.

## ArgsRegistry

``ArgsRegistry`` (a ``dict[str, Argument]``) is the unified interface that
ODE/PDE callables receive. It maps string keys to ``Argument`` instances.
Because ``Parameter`` extends ``Argument``, the callable is agnostic to
whether a value is fixed or being learned — it just calls
``args["key"](x)`` and gets a tensor back.

## InferredContext

``InferredContext`` is created automatically during data loading. It holds:

- **domain**: an N-dimensional ``Domain`` inferred from the training
  coordinates (bounds and step sizes).
- **validation**: resolved ground-truth functions for parameter comparison.

The context is injected into the ``Problem`` (and transitively into each
``Constraint``) before training starts. Constraints can override
``inject_context()`` to capture domain-specific information — for example,
``ICConstraint`` reads ``domain.x0`` to know where to enforce initial
conditions.

## Collocation strategies

Collocation points are the unsupervised sample locations where the PDE/ODE
residual is minimized. The choice of sampling strategy affects convergence:

- **uniform**: deterministic Cartesian grid. Predictable, but scales poorly
  to high dimensions.
- **random**: uniform random sampling. Simple and dimension-agnostic.
- **latin_hypercube**: stratified random sampling with better space-filling
  coverage than pure random. Good default for most problems.
- **log_uniform_1d**: samples densely near the lower domain bound. Useful
  for 1-D problems where early dynamics matter most (e.g. epidemic models).
- **adaptive**: residual-weighted resampling that concentrates points where
  the current model has the largest residual. Requires a ``ResidualScorer``
  and the ``AdaptiveCollocationCallback`` to refresh points during training.

Select a strategy via ``TrainingDataConfig(collocation_sampler="...")``.
"""

from anypinn.core.config import (
    AdamConfig,
    CosineAnnealingConfig,
    EarlyStoppingConfig,
    GenerationConfig,
    IngestionConfig,
    LBFGSConfig,
    MLPConfig,
    PINNHyperparameters,
    ReduceLROnPlateauConfig,
    ScalarConfig,
    SMMAStoppingConfig,
    TrainingDataConfig,
)
from anypinn.core.context import InferredContext
from anypinn.core.dataset import DataCallback, PINNDataModule, PINNDataset
from anypinn.core.nn import (
    ArgsRegistry,
    Argument,
    Domain,
    Field,
    FieldsRegistry,
    Parameter,
    ParamsRegistry,
    build_criterion,
    get_activation,
)
from anypinn.core.problem import Constraint, Problem
from anypinn.core.samplers import (
    AdaptiveSampler,
    CollocationSampler,
    LatinHypercubeSampler,
    LogUniform1DSampler,
    RandomSampler,
    ResidualScorer,
    UniformSampler,
    build_sampler,
)
from anypinn.core.types import (
    LOSS_KEY,
    Activations,
    CollocationStrategies,
    Criteria,
    DataBatch,
    LogFn,
    Predictions,
    TrainingBatch,
)
from anypinn.core.validation import (
    ColumnRef,
    ResolvedValidation,
    ValidationRegistry,
    ValidationSource,
    resolve_validation,
)
from anypinn.lib.encodings import FourierEncoding, RandomFourierFeatures

__all__ = [
    "LOSS_KEY",
    "Activations",
    "AdamConfig",
    "AdaptiveSampler",
    "ArgsRegistry",
    "Argument",
    "CollocationSampler",
    "CollocationStrategies",
    "ColumnRef",
    "Constraint",
    "CosineAnnealingConfig",
    "Criteria",
    "DataBatch",
    "DataCallback",
    "Domain",
    "EarlyStoppingConfig",
    "Field",
    "FieldsRegistry",
    "FourierEncoding",
    "GenerationConfig",
    "InferredContext",
    "IngestionConfig",
    "LBFGSConfig",
    "LatinHypercubeSampler",
    "LogFn",
    "LogUniform1DSampler",
    "MLPConfig",
    "PINNDataModule",
    "PINNDataset",
    "PINNHyperparameters",
    "Parameter",
    "ParamsRegistry",
    "Predictions",
    "Problem",
    "RandomFourierFeatures",
    "RandomSampler",
    "ReduceLROnPlateauConfig",
    "ResidualScorer",
    "ResolvedValidation",
    "SMMAStoppingConfig",
    "ScalarConfig",
    "TrainingBatch",
    "TrainingDataConfig",
    "UniformSampler",
    "ValidationRegistry",
    "ValidationSource",
    "build_criterion",
    "build_sampler",
    "get_activation",
    "resolve_validation",
]
