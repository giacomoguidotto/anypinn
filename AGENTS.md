# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Keep this file in sync.** If you change the architecture, data flow, module boundaries, or user-facing APIs, update this file and `README.md` to reflect those changes. Future contributors and AI agents rely on this being accurate.

## Project Vision

PINN is a modular Python library for solving differential equations using Physics-Informed Neural Networks. It is designed to be **scalable, modular, and progressively complex**:

- **Core layer** (`anypinn.core`): Pure PyTorch. Defines the mathematical problem — constraints, fields, parameters — with zero opinions about training. Users can plug `Problem.training_loss()` into any training loop.
- **Lightning layer** (`anypinn.lightning`): Optional. Wraps the core in PyTorch Lightning for batteries-included training. Users who don't want to manage training loops use this.
- **Problems** (`anypinn.problems`): Generalized ODE constraint types (residuals, initial conditions, data matching), `ODEHyperparameters`, and `ODEInverseProblem`.
- **Catalog** (`anypinn.catalog`): Ready-made building blocks for specific ODE systems (SIR, SEIR, Damped Oscillator, Lotka-Volterra) — ODE functions, constants, and DataModules.

The library serves three user profiles:

| User                  | What they do                                   | What they touch                                    |
| --------------------- | ---------------------------------------------- | -------------------------------------------------- |
| **Experimenter**      | Run a known problem, tweak params, see results | `examples/`, config dataclasses                    |
| **Researcher**        | Define new physics/constraints                 | `anypinn.core` (Constraint, Problem), `anypinn.problems` |
| **Framework builder** | Custom training, novel architectures           | `anypinn.core` only, skip Lightning                   |

## Commands

```bash
# Setup
uv sync                      # Install dependencies

# Testing & Quality (via just)
just test                    # Run tests with coverage
just lint                    # Check code style
just fmt                     # Format code (isort + ruff)
just lint-fix                # Auto-fix linting issues
just check                   # ty type checking
just ci                      # Run lint + check + test

# Documentation
just docs                    # Build mkdocs
just docs-serve              # Serve docs locally

# Direct invocation
uv run pytest tests/                    # Run tests directly
uv run pytest tests/test_foo.py::test_bar  # Run single test
```

## Verification

**Always run `just ci` to verify your work.** This is the single command that runs lint, type checking, and the full test suite in one shot. Do not run individual tools (`ruff`, `ty`, `pytest`) separately — `just ci` is the canonical check.

```bash
just ci   # lint + type check + full test suite (must pass before committing)
```

## Commit Conventions

This project uses [Conventional Commits](https://www.conventionalcommits.org/) with semantic-release. Commit prefixes trigger automated releases:

- `fix:` — patch release (0.0.X)
- `feat:` — minor release (0.X.0)
- `feat!:` / `BREAKING CHANGE:` — major release (X.0.0)

**Do not `git commit --amend` after pushing to `main`.** Semantic-release tags the original commit SHA; amending creates a new SHA and orphans the tag, causing the next release to skip with "already been released."

## Architecture

### Module Map

```
src/anypinn/
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
├── problems/              ← GENERALIZED ODE ABSTRACTIONS, depends on core
│   ├── ode.py             ← ResidualsConstraint, ICConstraint, DataConstraint,
│   │                         ODECallable, ODEProperties, ODEHyperparameters,
│   │                         ODEInverseProblem, PredictDataFn
│   └── pde.py             ← DirichletBCConstraint, NeumannBCConstraint,
│                             BoundaryCondition, BCValueFn
│
├── catalog/               ← PROBLEM-SPECIFIC BUILDING BLOCKS, depends on problems
│   ├── sir.py             ← SIR/rSIR ODE functions, SIRInvDataModule, constants
│   ├── damped_oscillator.py ← DampedOscillatorDataModule, constants
│   ├── lotka_volterra.py  ← LotkaVolterraDataModule, constants
│   └── seir.py            ← SEIRDataModule, constants
│
├── cli/                   ← STANDALONE, no dependency on core/lightning
│   ├── app.py             ← Typer app, `create` command entry point
│   ├── _types.py          ← Template, DataSource enums
│   ├── _prompts.py        ← Interactive prompts (Rich + simple-term-menu)
│   ├── _renderer.py       ← Project scaffolding / file generation
│   └── templates/         ← One module per template (sir, seir, etc.)
│       ├── _base.py       ← Shared train.py boilerplate (Lightning + core)
│       ├── _blank.py
│       ├── _sir.py
│       ├── _seir.py
│       ├── _damped_oscillator.py
│       ├── _lotka_volterra.py
│       └── _custom.py
│
└── lib/
    ├── diff.py            ← Differential operators (grad, partial, laplacian, …)
    └── utils.py           ← General-purpose helpers
```

### Dependency Direction (Strict)

```
anypinn.lightning ──depends on──▶ anypinn.core
anypinn.problems  ──depends on──▶ anypinn.core
anypinn.catalog   ──depends on──▶ anypinn.problems + anypinn.core
anypinn.lightning ──does NOT depend on──▶ anypinn.problems or anypinn.catalog
anypinn.core      ──does NOT depend on──▶ anything in anypinn.*
anypinn.cli       ──does NOT depend on──▶ anything in anypinn.*
```

This means:

- `anypinn.core` can be used completely standalone
- `anypinn.lightning` only knows about core abstractions, never specific problems
- `anypinn.problems` builds on core abstractions but doesn't require Lightning
- `anypinn.catalog` provides problem-specific building blocks (ODE functions, DataModules)
- `anypinn.cli` is a pure scaffolding tool — it generates project files but imports nothing from the library at runtime
- Examples wire everything together

### Core Abstractions (`anypinn.core`)

- **`Problem`** (`problem.py`): `nn.Module` that aggregates constraints, fields, and parameters. Provides `training_loss(batch, log)` and `predict(batch)`. The central object users build.
- **`Constraint`** (`problem.py`): Abstract base class for loss terms. Each subclass defines a `loss(batch, criterion, log)` method returning a weighted loss tensor. Supports runtime context injection.
- **`Field`** (`nn.py`): MLP mapping coordinates to state variables (e.g., `time -> [S, I, R]`). Xavier-initialized, configurable architecture.
- **`Parameter`** (`nn.py`): Learnable scalar (`nn.Parameter`) or function-valued (small MLP) parameter. Both expose a `forward(x)` interface.
- **`Argument`** (`nn.py`): Non-trainable wrapper for fixed values or callables. `Parameter` inherits from this.
- **`InferredContext`** (`context.py`): Runtime context (domain bounds, resolved validation) created from training data and injected into constraints.
- **`PINNDataset`** (`dataset.py`): PyTorch Dataset combining labeled data + collocation points per batch. Configurable `data_ratio`.
- **`PINNDataModule`** (`dataset.py`): Abstract Lightning DataModule. Subclasses implement `gen_data()` and `gen_coll()`. Handles context creation and validation resolution.

### Configuration System (`anypinn.core.config`)

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
├── optimizer: AdamConfig | LBFGSConfig | None
├── scheduler: ReduceLROnPlateauConfig | CosineAnnealingConfig | None
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

### ODE Problem Pattern (`anypinn.problems`)

ODE problems are composed from three constraint types, unified by `ODEInverseProblem`:

```
ODEProperties
├── ode: ODECallable(x, y, args) → dy/dx
├── args: ArgsRegistry (fixed parameters)
└── y0: Tensor (initial conditions)
        │
        ▼
ODEInverseProblem (composes all three with MSELoss)
├── ResidualsConstraint  ← ||dy/dt - f(t,y)||²  (uses autograd)
├── ICConstraint         ← ||y(t0) - y0||²      (needs context for t0)
└── DataConstraint       ← ||prediction - data||²

ODEHyperparameters (extends PINNHyperparameters)
├── pde_weight, ic_weight, data_weight
```

### Class Hierarchies

```
nn.Module
├── Field           (MLP: coordinates → state variables)
├── Parameter       (learnable scalar or MLP)
└── Problem         (aggregates constraints + registries)
    └── ODEInverseProblem  (generic ODE inverse, in anypinn.problems)

Constraint (ABC)
├── ResidualsConstraint   (ODE residual loss)
├── ICConstraint          (initial condition loss)
├── DataConstraint        (data-matching loss)
├── DirichletBCConstraint (Dirichlet BC: u = g on boundary)
└── NeumannBCConstraint   (Neumann BC: du/dn = h on boundary)

pl.LightningDataModule
└── PINNDataModule (ABC)
    ├── SIRInvDataModule           (in anypinn.catalog.sir)
    ├── DampedOscillatorDataModule (in anypinn.catalog.damped_oscillator)
    ├── LotkaVolterraDataModule    (in anypinn.catalog.lotka_volterra)
    └── SEIRDataModule             (in anypinn.catalog.seir)

pl.LightningModule
└── PINNModule
```

## Extending the Library

### Adding a New Problem

1. Define the ODE: function matching `ODECallable` protocol
2. Create the data module: `class MyDataModule(PINNDataModule)` implementing `gen_data()` and `gen_coll()` in `anypinn/catalog/`
3. Use `ODEInverseProblem` with `ODEHyperparameters` (or subclass if you need extra hyperparameters)
4. Wire it up in a training script (see `examples/`)

### Adding a New Constraint Type

1. Subclass `Constraint` from `anypinn.core.problem`
2. Implement `loss(batch, criterion, log)` returning a weighted loss tensor
3. Optionally override `inject_context()` if you need domain information

### Adding a New Problem Domain (e.g., PDEs)

1. Create a new module under `anypinn/problems/` (e.g., `pde.py`)
2. Define domain-specific constraint subclasses
3. Define a properties dataclass (like `ODEProperties`)
4. Keep it dependent only on `anypinn.core`
5. Add problem-specific building blocks in `anypinn/catalog/`

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

## Bootstrap CLI (`anypinn create`)

Scaffolding tool (like `npx create-next-app`) that generates a complete, runnable project:

```bash
anypinn create my-project [--template sir|seir|damped-oscillator|lotka-volterra|custom|blank]
                          [--data synthetic|csv]
                          [--lightning|--no-lightning]
```

Without flags the CLI runs interactively. Generated files:

- `pyproject.toml` — dependencies (conditionally includes `lightning`, `torchdiffeq`)
- `ode.py` — ODE function stub matching the `ODECallable` protocol
- `config.py` — hyperparameters with sensible defaults
- `train.py` — full training script (Lightning `Trainer` or plain PyTorch loop)
- `data/` — data directory

Implemented with Typer (`anypinn.cli`). Entry point: `anypinn.cli:app`.

## Code Style

- Line length: 99
- Ruff linter with rules: F, E, I, N, UP, RUF, B, C4, ISC, PIE, PT, PTH, SIM, TID
- Absolute imports only (no relative imports)
- ty type checking
- All config dataclasses: `@dataclass(frozen=True, kw_only=True)`
- Protocols for callable interfaces (`ODECallable`, `LogFn`, `PredictDataFn`)
- Registries (typed dicts) for flexible composition over rigid inheritance

## Examples

Located in `examples/`, each is a self-contained training script:

- `exponential_decay/exponential_decay.py` — **Minimal core-only example.** No Lightning. Learns decay rate k in dy/dt = -ky.
- `sir_inverse/sir_inverse.py` — Full SIR epidemic model
- `sir_inverse/reduced_sir_inverse.py` — Reduced SIR (single equation)
- `sir_inverse/hospitalized_sir_inverse.py` — SIR with hospitalization
- `damped_oscillator/damped_oscillator.py` — Damped harmonic oscillator
- `lotka_volterra/lotka_volterra.py` — Predator-prey dynamics
- `seir_inverse/seir_inverse.py` — SEIR epidemic model

All follow the same pattern: config → data module → fields/params → problem → PINNModule → Trainer.
