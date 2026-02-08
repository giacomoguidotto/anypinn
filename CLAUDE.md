# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Keep this file in sync.** If you change the architecture, data flow, module boundaries, or user-facing APIs, update this file and `README.md` to reflect those changes. Future contributors and AI agents rely on this being accurate.

## Project Vision

PINN is a modular Python library for solving differential equations using Physics-Informed Neural Networks. It is designed to be **scalable, modular, and progressively complex**:

- **Core layer** (`pinn.core`): Pure PyTorch. Defines the mathematical problem — constraints, fields, parameters — with zero opinions about training. Users can plug `Problem.training_loss()` into any training loop.
- **Lightning layer** (`pinn.lightning`): Optional. Wraps the core in PyTorch Lightning for batteries-included training. Users who don't want to manage training loops use this.
- **Problems** (`pinn.problems`): Ready-made constraint sets (ODE residuals, initial conditions, data matching) and full problem implementations (SIR, etc.) that users can pick up and run.

The library serves three user profiles:

| User                  | What they do                                   | What they touch                                    |
| --------------------- | ---------------------------------------------- | -------------------------------------------------- |
| **Experimenter**      | Run a known problem, tweak params, see results | `examples/`, config dataclasses                    |
| **Researcher**        | Define new physics/constraints                 | `pinn.core` (Constraint, Problem), `pinn.problems` |
| **Framework builder** | Custom training, novel architectures           | `pinn.core` only, skip Lightning                   |

## Commands

```bash
# Setup
uv sync                      # Install dependencies

# Testing & Quality (via nox)
uv run nox -s test           # Run tests (multi-Python, requires 100% coverage)
uv run nox -s lint           # Check code style
uv run nox -s fmt            # Format code (isort + ruff)
uv run nox -s lint_fix       # Auto-fix linting issues
uv run nox -s type_check     # MyPy strict type checking

# Documentation
uv run nox -s docs           # Build mkdocs
uv run nox -s docs_serve     # Serve docs locally

# Direct invocation
pytest tests/                # Run tests directly
pytest tests/test_foo.py::test_bar  # Run single test
```

## Architecture

### Module Map

```
src/pinn/
├── core/                  ← STANDALONE, pure PyTorch
│   ├── problem.py         ← Problem + Constraint (ABC)
│   ├── nn.py              ← Field (MLP), Parameter, Argument, Domain1D
│   ├── config.py          ← All config dataclasses
│   ├── context.py         ← InferredContext (runtime domain info)
│   ├── dataset.py         ← PINNDataset, PINNDataModule (ABC), DataCallback
│   ├── types.py           ← Type aliases, protocols (LogFn, TrainingBatch, etc.)
│   └── validation.py      ← ValidationRegistry, ColumnRef, resolution
│
├── lightning/             ← OPTIONAL, depends on core
│   ├── module.py          ← PINNModule (LightningModule wrapper)
│   └── callbacks.py       ← SMMAStopping, FormattedProgressBar,
│                             PredictionsWriter, DataScaling
│
├── problems/              ← REUSABLE TEMPLATES, depends on core
│   ├── ode.py             ← ResidualsConstraint, ICConstraint, DataConstraint,
│   │                         ODECallable protocol, ODEProperties
│   └── sir_inverse.py     ← SIR/rSIR ODE definitions, SIRInvProblem,
│                             SIRInvDataModule, SIRInvHyperparameters
│
└── lib/
    └── utils.py           ← General-purpose helpers
```

### Dependency Direction (Strict)

```
pinn.lightning ──depends on──▶ pinn.core
pinn.problems  ──depends on──▶ pinn.core
pinn.lightning ──does NOT depend on──▶ pinn.problems
pinn.core      ──does NOT depend on──▶ anything in pinn.*
```

This means:

- `pinn.core` can be used completely standalone
- `pinn.lightning` only knows about core abstractions, never specific problems
- `pinn.problems` builds on core abstractions but doesn't require Lightning
- Examples wire everything together

### Core Abstractions (`pinn.core`)

- **`Problem`** (`problem.py`): `nn.Module` that aggregates constraints, fields, and parameters. Provides `training_loss(batch, log)` and `predict(batch)`. The central object users build.
- **`Constraint`** (`problem.py`): Abstract base class for loss terms. Each subclass defines a `loss(batch, criterion, log)` method returning a weighted loss tensor. Supports runtime context injection.
- **`Field`** (`nn.py`): MLP mapping coordinates to state variables (e.g., `time -> [S, I, R]`). Xavier-initialized, configurable architecture.
- **`Parameter`** (`nn.py`): Learnable scalar (`nn.Parameter`) or function-valued (small MLP) parameter. Both expose a `forward(x)` interface.
- **`Argument`** (`nn.py`): Non-trainable wrapper for fixed values or callables. `Parameter` inherits from this.
- **`InferredContext`** (`context.py`): Runtime context (domain bounds, resolved validation) created from training data and injected into constraints.
- **`PINNDataset`** (`dataset.py`): PyTorch Dataset combining labeled data + collocation points per batch. Configurable `data_ratio`.
- **`PINNDataModule`** (`dataset.py`): Abstract Lightning DataModule. Subclasses implement `gen_data()` and `gen_coll()`. Handles context creation and validation resolution.

### Configuration System (`pinn.core.config`)

All configs are frozen dataclasses with `kw_only=True`:

```
PINNHyperparameters
├── lr: float
├── training_data: IngestionConfig | GenerationConfig
│   ├── batch_size, data_ratio, collocations
│   ├── (Ingestion): df_path, x_column, y_columns, x_transform
│   └── (Generation): x, noise_level, args_to_train
├── fields_config: MLPConfig
│   └── in_dim, out_dim, hidden_layers, activation, output_activation, encode
├── params_config: MLPConfig | ScalarConfig
├── scheduler: SchedulerConfig | None
├── early_stopping: EarlyStoppingConfig | None
└── smma_stopping: SMMAStoppingConfig | None
```

### Registry Pattern

The library uses typed dictionaries as registries for flexible composition:

```python
FieldsRegistry  = dict[str, Field]              # State variable networks
ParamsRegistry  = dict[str, Parameter]           # Learnable parameters
ArgsRegistry    = dict[str, Argument]            # Fixed/callable parameters
ValidationRegistry = dict[str, ValidationSource] # Ground truth references
```

### Data Flow: Training

```
Data Source (CSV or synthetic generation)
        │
        ▼
PINNDataModule
├── load_data() / gen_data()       ← produce (x, y) pairs
├── gen_coll()                     ← produce collocation points
├── DataCallback.transform_data()  ← optional transforms (e.g., DataScaling)
└── setup()
        │
        ▼
InferredContext
├── domain: Domain1D (x0, x1, dx)
└── validation: ResolvedValidation (name → callable)
        │
        ▼
PINNDataset → DataLoader (7 workers, persistent)
├── Yields: ((x_data, y_data), x_coll) per batch
├── Data sampled without replacement
└── Collocation sampled with replacement
        │
        ▼
PINNModule.training_step() → Problem.training_loss()
├── Each Constraint.loss() computes its term
└── Returns Σ weighted losses
        │
        ▼
Adam optimizer + optional ReduceLROnPlateau scheduler
```

### Data Flow: Prediction

```
PINNModule.predict_step(batch)
        │
        ▼
Problem.predict(batch)
├── Field(x) → unscaled state variables
├── Parameter(x) → learned parameter values
└── true_values(x) → ground truth from validation
        │
        ▼
Predictions = ((x, y_pred), params_dict, true_values_dict | None)
```

### ODE Problem Pattern (`pinn.problems`)

ODE problems are composed from three constraint types:

```
ODEProperties
├── ode: ODECallable(x, y, args) → dy/dx
├── args: ArgsRegistry (fixed parameters)
└── y0: Tensor (initial conditions)
        │
        ▼
Constraints:
├── ResidualsConstraint  ← ||dy/dt - f(t,y)||²  (uses autograd)
├── ICConstraint         ← ||y(t0) - y0||²      (needs context for t0)
└── DataConstraint       ← ||prediction - data||²
```

### Class Hierarchies

```
nn.Module
├── Field           (MLP: coordinates → state variables)
├── Parameter       (learnable scalar or MLP)
└── Problem         (aggregates constraints + registries)
    └── SIRInvProblem, etc.

Constraint (ABC)
├── ResidualsConstraint   (ODE residual loss)
├── ICConstraint          (initial condition loss)
└── DataConstraint        (data-matching loss)

pl.LightningDataModule
└── PINNDataModule (ABC)
    └── SIRInvDataModule, etc.

pl.LightningModule
└── PINNModule
```

## Extending the Library

### Adding a New Problem

1. Define hyperparameters: `class MyHyperparameters(PINNHyperparameters)` with custom weights
2. Define the ODE: function matching `ODECallable` protocol
3. Create the problem: `class MyProblem(Problem)` composing constraints from `ode.py`
4. Create the data module: `class MyDataModule(PINNDataModule)` implementing `gen_data()` and `gen_coll()`
5. Wire it up in a training script (see `examples/`)

### Adding a New Constraint Type

1. Subclass `Constraint` from `pinn.core.problem`
2. Implement `loss(batch, criterion, log)` returning a weighted loss tensor
3. Optionally override `inject_context()` if you need domain information

### Adding a New Problem Domain (e.g., PDEs)

1. Create a new module under `pinn/problems/` (e.g., `pde.py`)
2. Define domain-specific constraint subclasses
3. Define a properties dataclass (like `ODEProperties`)
4. Keep it dependent only on `pinn.core`

### Using Core Without Lightning

```python
# Build problem
problem = MyProblem(constraints=[...], fields={...}, params={...})
problem.inject_context(context)

# Your own training loop
optimizer = torch.optim.Adam(problem.parameters(), lr=1e-3)
for batch in your_dataloader:
    optimizer.zero_grad()
    loss = problem.training_loss(batch, log=your_log_fn)
    loss.backward()
    optimizer.step()
```

## Future: Bootstrap CLI (`pinn create`)

Planned scaffolding tool (like `npx create-next-app`) to generate boilerplate:

- Pick from supported templates (SIR, SEIR, Lotka-Volterra, Damped Oscillator, ...)
- Define a new ODE problem interactively
- Start from a blank project
- Choose data source (synthetic or CSV)
- Choose whether to include Lightning wrapper

This will be implemented using Typer (already a dependency via `src/fact/cli.py`).

## Code Style

- Line length: 99
- Ruff linter with rules: F, E, I, N, UP, RUF, B, C4, ISC, PIE, PT, PTH, SIM, TID
- Absolute imports only (no relative imports)
- MyPy strict mode with exhaustive-match enabled
- All config dataclasses: `@dataclass(frozen=True, kw_only=True)`
- Protocols for callable interfaces (`ODECallable`, `LogFn`, `PredictDataFn`)
- Registries (typed dicts) for flexible composition over rigid inheritance

## Examples

Located in `examples/`, each is a self-contained training script:

- `sir_inverse/sir_inverse.py` — Full SIR epidemic model
- `sir_inverse/reduced_sir_inverse.py` — Reduced SIR (single equation)
- `sir_inverse/hospitalized_sir_inverse.py` — SIR with hospitalization
- `damped_oscillator/damped_oscillator.py` — Damped harmonic oscillator
- `lotka_volterra/lotka_volterra.py` — Predator-prey dynamics
- `seir_inverse/seir_inverse.py` — SEIR epidemic model

All follow the same pattern: config → data module → fields/params → problem → PINNModule → Trainer.
