---
icon: material/math-integral
---

# AnyPINN User Guide

A modular, extensible Python library for solving differential equations using
Physics-Informed Neural Networks (PINNs). Built for scalability — from one-click
experiments to fully custom problem definitions.

## Installation

First, [install `uv`](https://docs.astral.sh/uv/getting-started/installation):

=== "macOS and Linux"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

Then install `anypinn` and its dependencies:

```bash
uv sync
```

## Quick Start

Run one of the included examples to see AnyPINN in action:

```bash
cd examples/sir_inverse
python sir_inverse.py
```

### Available Examples

| Example | Description |
|:--------|:------------|
| `sir_inverse/sir_inverse.py` | Full SIR epidemic model |
| `sir_inverse/reduced_sir_inverse.py` | Reduced SIR (single equation) |
| `sir_inverse/hospitalized_sir_inverse.py` | SIR with hospitalization |
| `damped_oscillator/damped_oscillator.py` | Damped harmonic oscillator |
| `lotka_volterra/lotka_volterra.py` | Predator-prey dynamics |
| `seir_inverse/seir_inverse.py` | SEIR epidemic model |

## Architecture Overview

AnyPINN is split into independent layers that you can use separately:

- **`anypinn.core`** — Pure PyTorch. Defines the mathematical problem with zero opinions
  about training.
- **`anypinn.lightning`** — Optional. Wraps core in PyTorch Lightning for batteries-included
  training.
- **`anypinn.problems`** — Generalized ODE constraint types (residuals, initial conditions,
  data matching).
- **`anypinn.catalog`** — Ready-made building blocks for specific ODE systems (SIR, SEIR,
  Damped Oscillator, Lotka-Volterra).

!!! tip

    You can use `anypinn.core` completely standalone — plug `Problem.training_loss()` into
    any training loop you like (plain PyTorch, Hugging Face Accelerate, etc.).

## Defining a New Problem

### 1. Define Your ODE

Write a function matching the `ODECallable` protocol:

```python
from torch import Tensor

from anypinn.core.types import ArgsRegistry


def my_ode(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    # Return dy/dx
    ...
```

### 2. Build the Problem

Compose constraints into a problem:

```python
from anypinn.problems.ode import (
    DataConstraint,
    ODEProperties,
    ResidualsConstraint,
)

problem = MyProblem(
    constraints=[
        ResidualsConstraint(field, ode_props, weight=hp.pde_weight),
        DataConstraint(predict_fn, weight=hp.data_weight),
    ],
    fields={"u": field},
    params={"k": param},
)
```

### 3. Train

=== "With Lightning"

    ```python
    import lightning as pl

    from anypinn.lightning.module import PINNModule

    module = PINNModule(problem, hp)
    trainer = pl.Trainer(max_epochs=50000)
    trainer.fit(module, datamodule=dm)
    ```

=== "Plain PyTorch"

    ```python
    import torch

    optimizer = torch.optim.Adam(problem.parameters(), lr=1e-3)
    for batch in dataloader:
        optimizer.zero_grad()
        loss = problem.training_loss(batch, log=my_log_fn)
        loss.backward()
        optimizer.step()
    ```

## Development

### Commands

```bash
uv run nox -s test           # Run tests (100% coverage required)
uv run nox -s lint           # Check code style
uv run nox -s fmt            # Format code (isort + ruff)
uv run nox -s lint_fix       # Auto-fix linting issues
uv run nox -s type_check     # MyPy strict type checking
uv run nox -s docs           # Build documentation
uv run nox -s docs_serve     # Serve docs locally
```
