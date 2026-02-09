# AnyPINN

[![GitHub Actions][github-actions-badge]](https://github.com/johnthagen/python-blueprint/actions)
[![uv][uv-badge]](https://github.com/astral-sh/uv)
[![Nox][nox-badge]](https://github.com/wntrblm/nox)
[![Ruff][ruff-badge]](https://github.com/astral-sh/ruff)
[![Type checked with mypy][mypy-badge]](https://mypy-lang.org/)

[github-actions-badge]: https://github.com/johnthagen/python-blueprint/actions/workflows/ci.yml/badge.svg
[uv-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json
[nox-badge]: https://img.shields.io/badge/%F0%9F%A6%8A-Nox-D85E00.svg
[ruff-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[mypy-badge]: https://www.mypy-lang.org/static/mypy_badge.svg

A modular, extensible Python library for solving differential equations using Physics-Informed Neural Networks (PINNs). Built for scalability — from one-click experiments to fully custom problem definitions.

## Philosophy

AnyPINN is designed around two principles:

1. **Separation of concerns.** The mathematical problem definition is completely decoupled from the training engine. You can use one without the other.
2. **Progressive complexity.** Start simple, go deep only when you need to.

This means the library serves three types of users:

| User                  | Goal                                               | How                                                                    |
| --------------------- | -------------------------------------------------- | ---------------------------------------------------------------------- |
| **Experimenter**      | Run a known problem, tweak parameters, see results | Pick a built-in problem, change config, press start                    |
| **Researcher**        | Define a new problem with custom physics           | Implement `Constraint` and `Problem`, use the provided training engine |
| **Framework builder** | Custom training loops, novel architectures         | Use the core abstractions directly, skip Lightning entirely            |

## Architecture

The library is split into two independent layers:

```mermaid
graph TD
    EXP["Your Experiment<br/><i>examples/ or your own script</i>"]

    EXP --> LIT
    EXP --> CORE

    subgraph LIT["anypinn.lightning (optional)"]
        direction LR
        L1[PINNModule]
        L2[Callbacks]
        L3[PINNDataModule]
        L4[PredictionsWriter]
    end

    subgraph CORE["anypinn.core (standalone)"]
        direction LR
        C1["Problem — Constraint (ABC)"]
        C2["Field — MLP networks"]
        C3["Parameter — learnable scalars"]
        C4["Config — dataclass configs"]
        C5["Context — runtime domain info"]
        C6["Validation — ground truth refs"]
    end

    LIT -->|depends on| CORE
```

### Core (`anypinn.core`) — The Math Layer

The core is a pure PyTorch library. It defines what a PINN problem _is_, with no opinions about how you train it.

- **`Problem`** — Aggregates constraints, fields, and parameters. Provides `training_loss()` and `predict()`.
- **`Constraint`** (abstract) — A single loss term. Subclass it to define any physics equation, boundary condition, or data-matching loss.
- **`Field`** — An MLP that maps input coordinates to state variables (e.g., `t -> [S, I, R]`).
- **`Parameter`** — A learnable scalar or function-valued parameter (e.g., `beta` in an SIR model).
- **`InferredContext`** — Runtime information (domain bounds, validation data) extracted from training data and injected into constraints.

You can use `Problem.training_loss()` inside any training loop — plain PyTorch, Hugging Face Accelerate, or anything else.

### Lightning (`anypinn.lightning`) — The Training Engine (Optional)

A thin wrapper that plugs a `Problem` into PyTorch Lightning. Use it when you want batteries-included training with minimal boilerplate:

- **`PINNModule`** — Wraps a `Problem` as a `LightningModule`. Handles optimizer setup, context injection, and prediction.
- **`PINNDataModule`** — Abstract data module that manages data loading, collocation point generation, and context creation.
- **`Callbacks`** — SMMA-based early stopping, formatted progress bars, prediction writers, data scaling.

### Problems (`anypinn.problems`) — Ready-Made Templates

Pre-built constraint sets for common problem types:

- **ODE layer** (`ode.py`): `ResidualsConstraint`, `ICConstraint`, `DataConstraint` — covers most ODE inverse problems out of the box.
- **SIR Inverse** (`sir_inverse.py`): Full and reduced SIR model implementations.

## Data Flow

### Training

```mermaid
graph TD
    DS["Data Source<br/><i>CSV or synthetic</i>"]
    DM["PINNDataModule"]
    DM1["load_data() / gen_data() — produce (x, y) pairs"]
    DM2["gen_coll() — produce collocation points"]
    DM3["DataCallback.transform_data() — optional scaling"]
    DM4["setup()"]
    CTX["InferredContext<br/><i>domain bounds, resolved validation</i>"]
    DSET["PINNDataset<br/><i>batches of labeled data + collocation points</i>"]
    STEP["PINNModule.training_step(batch)"]
    LOSS["Problem.training_loss(batch)"]
    C1["Constraint₁.loss() — ODE residuals"]
    C2["Constraint₂.loss() — initial conditions"]
    C3["Constraint₃.loss() — data matching"]
    BP["Σ weighted losses → backprop → Adam + optional scheduler"]

    DS --> DM
    DM --- DM1
    DM --- DM2
    DM --- DM3
    DM --- DM4
    DM --> CTX --> DSET --> STEP --> LOSS
    LOSS --> C1
    LOSS --> C2
    LOSS --> C3
    C1 --> BP
    C2 --> BP
    C3 --> BP
```

### Prediction

```mermaid
graph TD
    PS["PINNModule.predict_step(batch)"]
    PP["Problem.predict(batch)"]
    F["Field(x) → state variables (unscaled)"]
    P["Parameter(x) → learned parameters"]
    T["true_values(x) → ground truth (if available)"]
    OUT["((x, y_pred), params_dict, true_values_dict)"]

    PS --> PP
    PP --> F
    PP --> P
    PP --> T
    F --> OUT
    P --> OUT
    T --> OUT
```

## Getting Started

### Installation

```bash
uv sync
```

### Run an Example

```bash
cd examples/sir_inverse
python sir_inverse.py
```

### Implement a New Problem

1. **Define your ODE** as a callable matching the `ODECallable` protocol:

```python
def my_ode(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    # Return dy/dx
    ...
```

2. **Configure hyperparameters**:

```python
@dataclass(frozen=True, kw_only=True)
class MyHyperparameters(PINNHyperparameters):
    pde_weight: float = 1.0
    data_weight: float = 1.0
```

3. **Build the problem** from constraints:

```python
problem = MyProblem(
    constraints=[
        ResidualsConstraint(field, ode_props, weight=hp.pde_weight),
        DataConstraint(predict_fn, weight=hp.data_weight),
    ],
    fields={"u": field},
    params={"k": param},
)
```

4. **Train** with Lightning or your own loop:

```python
# With Lightning
module = PINNModule(problem, hp)
trainer = pl.Trainer(max_epochs=50000)
trainer.fit(module, datamodule=dm)

# Or plain PyTorch
for batch in dataloader:
    loss = problem.training_loss(batch, log=my_log_fn)
    loss.backward()
    optimizer.step()
```

See `examples/` for complete implementations:

- `sir_inverse/` — SIR epidemic model (full, reduced, hospitalized variants)
- `damped_oscillator/` — Damped harmonic oscillator
- `lotka_volterra/` — Predator-prey dynamics
- `seir_inverse/` — SEIR epidemic model

## Future: Bootstrap CLI (`anypinn create`)

Planned: a scaffolding tool inspired by `npx create-next-app` that lets you bootstrap a new PINN project interactively:

```
$ anypinn create my-project

? Choose a starting point:
  > From a template (SIR, SEIR, Lotka-Volterra, Damped Oscillator, ...)
    Define a new ODE problem
    Blank project

? Select training data source:
  > Generate synthetic data
    Load from CSV

? Include Lightning training wrapper? (Y/n)

Creating my-project/...
  my_problem.py     — problem definition
  train.py          — training script
  config.py         — hyperparameters
  data/             — data directory
Done.
```

This will lower the barrier for experimenters who want to try a known problem with their own data without writing boilerplate.

## Development

### Tooling

| Tool                                      | Purpose                |
| ----------------------------------------- | ---------------------- |
| [uv](https://github.com/astral-sh/uv)     | Dependency management  |
| [Nox](https://github.com/wntrblm/nox)     | Task automation        |
| [Ruff](https://github.com/astral-sh/ruff) | Linting and formatting |
| [pytest](https://docs.pytest.org/)        | Testing                |
| [mypy](https://mypy-lang.org/)            | Strict type checking   |

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

## Contributing

When contributing:

- Follow the existing code style (Ruff, line length 99, absolute imports only)
- Keep the two-layer separation: core stays pure PyTorch, Lightning stays optional
- If you change the architecture or data flow, update both `CLAUDE.md` and this README to reflect the changes
