"""Lightning integration for PINN training.

This module wraps ``anypinn.core`` abstractions into PyTorch Lightning
components: a training module, stopping criteria, and data callbacks.

## PINNModule

``PINNModule`` is a thin ``LightningModule`` that delegates to a ``Problem``
instance. It handles optimizer/scheduler configuration, context injection at
fit start, and prediction output formatting. You rarely need to subclass it
— all physics live in the ``Problem`` and its ``Constraint`` list.

## Data scaling

When ODE/PDE state variables span very different magnitudes (e.g. the Lorenz
system where variables reach ~40), raw values can destabilize training.
``DataScaling`` is a ``DataCallback`` (not a Lightning Callback) that
rescales data **before** the dataset is constructed:

- **x** is normalized to [0, 1].
- **y** is multiplied by per-series scale factors you provide.

Pass it via ``PINNDataModule(callbacks=[DataScaling(y_scale=...)])``.
Validation functions are automatically rescaled so that logged validation
losses remain comparable.

## Stopping criteria

Two stopping strategies are available:

- **SMMAStopping**: monitors the Smoothed Moving Average of the loss and
  stops when relative improvement over a lookback window drops below a
  threshold. This is the default in most catalog examples — it adapts to
  the loss trajectory rather than requiring a fixed patience count.
- **Lightning's built-in EarlyStopping**: monitors a metric and stops after
  ``patience`` epochs without improvement. Simpler to reason about, but
  less tolerant of noisy loss curves common in PINN training.

Use ``SMMAStopping`` when loss decreases slowly and erratically (typical for
PINNs). Use ``EarlyStopping`` when loss curves are smooth and you want a
hard patience bound.

## Adaptive collocation

``AdaptiveCollocationCallback`` resamples collocation points every N epochs
using the current model weights. It requires the data module to be
configured with ``collocation_sampler="adaptive"`` and a ``ResidualScorer``.
"""

from anypinn.lightning.callbacks import (
    AdaptiveCollocationCallback,
    FormattedProgressBar,
    PredictionsWriter,
    SMMAStopping,
)
from anypinn.lightning.module import PINNModule

__all__ = [
    "AdaptiveCollocationCallback",
    "FormattedProgressBar",
    "PINNModule",
    "PredictionsWriter",
    "SMMAStopping",
]
