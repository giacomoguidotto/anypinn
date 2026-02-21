---
icon: material/scale-balance
---

# Design Rationale & Future Work

This document addresses two questions: **why AnyPINN exists** given the breadth of existing PINN
libraries, and **what remains to be built** for the library to reach its full intended scope.

---

## 1. Positioning Against Existing Libraries

The Physics-Informed Neural Network ecosystem already contains a dozen libraries spanning two major
frameworks. The question is therefore not "does a gap exist?" but "which tradeoffs does each library
make, and which tradeoffs are the right ones for which users?".

The libraries considered here include PyTorch-based tools (NeuroDiffEq, IDRLNet, NVIDIA Modulus,
PINA) and TensorFlow-based tools (DeepXDE, TensorDiffEq, SciANN, PyDEns, Elvet, NVIDIA SimNet).

### 1.1 The Framework Ecosystem Problem

More than half of the existing libraries — DeepXDE (in its original form), TensorDiffEq, SciANN,
PyDEns, Elvet, and NVIDIA SimNet — are TensorFlow-based. This is a meaningful architectural
liability. PyTorch has become the de-facto standard for ML research: its share of papers, tooling,
and downstream ecosystem (Lightning, Hugging Face, einops, triton, `torch.compile`) dwarfs
TensorFlow's in the research space. Directing a researcher to a TensorFlow-based PINN library in
2025 adds framework friction that has nothing to do with the science.

AnyPINN is pure PyTorch from the ground up. Its core primitives (`Field`, `Parameter`, `Problem`,
`Constraint`) are all `nn.Module` subclasses. This means the library participates naturally in the
entire PyTorch ecosystem: mixed precision, `torch.compile`, distributed data parallel, gradient
checkpointing — none of these require library-level support because the library never wraps the
training loop.

### 1.2 The Training Engine Coupling Problem

Every major competitor couples physics problem definition to its own training loop:

| Library | Coupling mechanism |
|---|---|
| NeuroDiffEq | `solve_ivp`-style API owns training |
| DeepXDE | `Model.train()` is not optional |
| IDRLNet | Computational graph node system owns training |
| NVIDIA Modulus | Full platform: owns Trainer, distributed, logging |
| PINA | Own `Trainer` wrapping Lightning |

The consequence is that when the training ecosystem evolves — new optimizers, new distributed
strategies, new precision formats — users must wait for the library to support them or write
wrappers around the library's own abstractions.

AnyPINN takes a different position: **training is not the library's business.** `Problem` is an
`nn.Module`. `Problem.training_loss(batch, log)` takes a batch and returns a scalar tensor. The
contract ends there. A minimal training loop looks exactly like any other PyTorch training loop:

```python
optimizer = torch.optim.Adam(problem.parameters(), lr=1e-3)
for batch in dataloader:
    optimizer.zero_grad()
    loss = problem.training_loss(batch, log=my_log_fn)
    loss.backward()
    optimizer.step()
```

PyTorch Lightning is available as an *optional* layer (`anypinn.lightning`) for users who want
batteries-included training. It is not required. This is the right default for a research library:
researchers frequently need non-standard training procedures, and a library that owns the training
loop forces workarounds.

### 1.3 Inverse Problems as a First-Class Abstraction

Most PINN libraries were designed for **forward problems**: given a known PDE and boundary or
initial conditions, find the solution field `u(x, t)`. Inverse problems — recover unknown
parameters from partial observations — are typically supported as extensions or workarounds.

AnyPINN treats the inverse problem as the primary use case. This is reflected throughout the type
system:

**`Parameter` is a top-level type, not a configuration flag.** It can represent a learned scalar
(e.g., a fixed transmission rate `β`) or a function-valued learnable parameter (e.g., a
time-varying `β(t)` backed by a small MLP). Both expose the same `forward(x)` interface and
integrate transparently into the `ArgsRegistry`, so the ODE function cannot distinguish between a
fixed argument and a learnable one. Promoting a fixed parameter to a learnable one requires
changing one config line, not restructuring the problem definition.

**`ValidationRegistry` provides ground-truth tracking for recovered parameters.** A CSV column can
be bound to a parameter name at construction time. The library then logs the MSE between the
recovered parameter and the known ground truth at every training step. This workflow — recovering
parameters and continuously comparing them against a known reference — is central to inverse
problems and is not present as a first-class feature in any of the listed alternatives.

**`ArgsRegistry` unifies fixed and learnable arguments.** The ODE callable receives an
`ArgsRegistry` — a typed dict of `Argument` instances. `Parameter` inherits from `Argument`, so
the same ODE function works whether the parameters are fixed or learned. This composability is
absent from libraries that distinguish the two at the problem-definition level.

### 1.4 Progressive Complexity and the Three-Audience Model

Research libraries tend to optimize for one audience. Modulus targets HPC engineers running 3D
CFD. NeuroDiffEq targets researchers who want a declarative ODE/PDE API. Neither serves the user
who wants to start with a known problem and progressively take control of more layers.

AnyPINN is explicitly designed for three audiences, and the architecture enforces the separation:

```
anypinn.catalog      ← Experimenter: pick a known problem, change config, run
anypinn.problems     ← Researcher: define new physics, use provided constraints
anypinn.core         ← Framework builder: skip Lightning, own your training loop
anypinn.lightning    ← Optional at every level
```

The strict dependency direction (`catalog → problems → core`, `lightning → core`) ensures that each
layer is usable without the ones above it. A user who only needs `anypinn.core` takes no
dependency on Lightning, no dependency on problem-specific constraint implementations, and no
dependency on the catalog. The core is 193 lines of `problem.py` and can be understood in a single
reading.

### 1.5 Developer Experience as a First-Class Concern

Research libraries are routinely distributed in a state that would be unacceptable in production
software: `assert` statements for validation, positional-argument constructors, no config
validation, no type stubs. These are not cosmetic complaints — they produce silent failures,
confusing errors, and codebases that are hard to extend.

AnyPINN addresses this with:

- **Typed frozen dataclasses for all configuration.** Every hyperparameter is a `@dataclass(frozen=True, kw_only=True)`. Configs are immutable, keyword-only, and introspectable. This is in contrast to libraries like NeuroDiffEq that pass hyperparameters as positional arguments, or DeepXDE that uses a mix of constructor arguments and setter methods.
- **Static type checking with `ty` in CI.** Type errors are caught before runtime.
- **Semantic versioning with automated releases.** Conventional Commits trigger patch/minor/major releases automatically. Users can pin versions with confidence.
- **Modern packaging.** `uv` for dependency management, `hatchling` with dynamic versioning from VCS tags, Ruff for linting and formatting.

### 1.6 The Bootstrap CLI

No library in the comparison set ships a project scaffolding tool. Every user faces the same
cold-start problem: before writing a single line of physics, they must learn the library API, set
up a project structure, wire together data loading, training, and logging.

`anypinn create my-project` resolves this:

```
$ anypinn create my-project

◇  Choose a starting point
│  SIR Epidemic Model

◇  Select training data source
│  Generate synthetic data

◇  Include Lightning training wrapper?
│  Yes

◇  Creating my-project/...
│  pyproject.toml   — project dependencies
│  ode.py           — mathematical definition
│  config.py        — training configuration
│  train.py         — execution script
│  data/            — data directory

●  Done! cd my-project && uv sync && uv run train.py
```

The generated project is fully runnable. The researcher's first meaningful action is reading and
modifying `ode.py` — not reading library documentation to understand what arguments
`PINNDataModule` needs. The `--lightning / --no-lightning` flag is particularly valuable: it
generates a training script matched to the user's chosen layer, teaching the two-layer architecture
by letting the user choose.

### 1.7 Honest Scope Limitations

The argument above holds within a specific scope. AnyPINN is not the right choice for:

- **PDE problems** (heat, wave, Navier-Stokes). The current architecture only supports 1D domains
  (`Domain1D`), and `PINNDataset` shape assertions block multi-dimensional inputs. DeepXDE and PINA
  are better choices for PDE problems today.
- **Large-scale 3D simulations on GPU clusters.** NVIDIA Modulus is purpose-built for this and
  has no peer in that space.
- **Users already productive in TensorFlow.** Switching frameworks for a library is rarely
  justified unless the problem demands it.

The library's primary justification is the **ODE inverse problem** workflow: recovering parameters
from partial observations of a dynamical system (epidemiological, mechanical, ecological). This is
where the `Parameter` / `ValidationRegistry` / `ArgsRegistry` design is most directly
valuable and most clearly unmatched by existing alternatives.

---

## 2. Future Work

The items below are drawn from the architecture audit. They are grouped by scope and ordered by
impact within each group. Scaling and performance items have all been resolved; what remains is
developer experience hardening and the PDE expansion track.

### 2.1 Developer Experience

These items address correctness and usability. None require architectural changes — they are
disciplined application of existing patterns.

#### D1 — Replace assertions with proper exceptions

**Files:** `core/dataset.py`, `core/nn.py`

All shape and value checks currently use `assert`, which Python strips with `-O`. A
`ValueError` or `TypeError` with a descriptive message (expected shape vs. actual shape) is the
correct mechanism. This is a small but high-impact change: a cryptic `AssertionError` with no
context is one of the most common causes of user confusion in numerical libraries.

#### D2 — Config validation in `__post_init__`

**File:** `core/config.py`

Hyperparameters like `lr`, `batch_size`, `data_ratio`, and `collocations` are never validated
at construction time. A negative learning rate or zero batch size currently surfaces only as a
cryptic runtime error deep in training. Adding `__post_init__` guards with clear messages (e.g.,
`"lr must be positive, got -1e-3"`) catches misconfiguration at the earliest possible point.

#### D3 — Registry key validation at construction

**Files:** `core/problem.py`, `problems/ode.py`

If the keys in `FieldsRegistry` don't match what the ODE callable expects, the error surfaces as
a shape mismatch or `KeyError` deep inside an autograd graph — the worst possible location to
debug. A builder or factory that validates registry keys against the ODE function signature at
construction time would catch this immediately.

#### D4 — Explicit shape contracts instead of implicit squeeze

**File:** `core/problem.py`

`f(x).squeeze(-1)` silently drops dimensions. If a `Field` accidentally outputs shape
`(N, 1, 1)`, `squeeze(-1)` produces `(N, 1)` — different from the `(N,)` expected downstream,
with no error until a later operation. Explicit `reshape` calls with documented shape contracts
are safer and self-documenting.

#### D5 — Interpolation strategy for non-integer `ColumnRef` indices

**File:** `core/validation.py`

The current `ColumnRef` resolution uses `x.round().to(torch.int32)` to index into a tensor of
ground-truth values. This only works correctly when `x` values are evenly-spaced integers. For
continuous or irregularly-sampled `x`, it silently returns wrong values. The fix requires either
a proper interpolation strategy (`torch.searchsorted` + linear interpolation) or an explicit
`x`-to-index mapping passed at construction time.

#### D6 — Cache CSV reads in `ColumnRef` resolution

**File:** `core/validation.py`

Each `ColumnRef` entry triggers a fresh `pd.read_csv()`. A problem with five validated
parameters reads the same CSV five times. The data frame should be read once and shared across all
`ColumnRef` instances that reference the same file.

#### D7 — Robust shape handling in `load_data` for 1-D `y`

**File:** `core/dataset.py`

The guard `if y.shape[1] != 1` raises an `IndexError` when `y` has shape `(N,)` (a 1-D tensor),
because `shape[1]` does not exist. This edge case is triggered by single-column CSV inputs that
haven't been explicitly reshaped. The fix is a `y.ndim` check before indexing `shape[1]`.

#### D8 — Minimal core-only example

**Files:** `examples/`

Every existing example uses the full Lightning stack with scaling, SMMA stopping, and custom
progress bars. A 20-30 line example using only `anypinn.core` with a plain PyTorch training loop
would substantially lower the onboarding barrier for users who want to understand the library from
the ground up before adopting Lightning.

---

### 2.2 PDE Expansion

These items represent a coherent track of work that would extend AnyPINN from ODE-only to full
PDE support. They have strict dependencies between them: PDE1, PDE2, and PDE5 are blockers for
everything else in this group. The items are listed in recommended implementation order.

!!! warning
    PDE1 + PDE2 + PDE5 must be resolved together before any other PDE work is useful. They are
    the three structural blockers.

#### PDE1 — Multi-dimensional domain abstraction (Critical blocker)

**File:** `core/nn.py`

`Domain1D` hard-codes a 1D interval `[x0, x1]` with scalar step `dx`. This assumption propagates
through `InferredContext`, `PINNDataModule.gen_coll()`, and `PINNDataset` shape assertions.

The required change is a `Domain` base class hierarchy:

```python
class Domain(ABC):
    @abstractmethod
    def sample(self, n: int) -> Tensor: ...

class Domain1D(Domain): ...
class Domain2D(Domain): ...   # rectangle [x0,x1] × [y0,y1]
class DomainND(Domain): ...   # n-dimensional hypercube
```

`InferredContext` should hold a `Domain`, and `gen_coll()` should accept and return
multi-dimensional tensors of shape `(M, d)`.

#### PDE2 — Boundary condition constraint types (Critical blocker)

**File:** `problems/ode.py` (new `problems/pde.py`)

The three existing constraint types (`ResidualsConstraint`, `ICConstraint`, `DataConstraint`) are
ODE-specific. PDEs require boundary condition constraint types in a new `anypinn.problems.pde`
module:

- `DirichletConstraint`: enforces `u(x) = g(x)` on boundary points
- `NeumannConstraint`: enforces `du/dn = h(x)` on boundary points
- `RobinConstraint`: enforces `αu + β(du/dn) = g(x)`
- Periodic BC support

Each should be a `Constraint` subclass receiving a boundary region sampler and a target function.

#### PDE5 — Relax 1D shape assertions (Critical blocker)

**File:** `core/dataset.py`

```python
assert x_data.shape[1] == 1, "x shape differs than (n, 1)."
assert self.coll.shape[1] == 1, "coll shape differs than (m, 1)."
```

These assertions block all multi-dimensional inputs. They should be generalized to
`shape[1] == d` where `d` is the spatial dimension inferred from the domain.

#### PDE3 — Higher-order and mixed derivative utilities

**File:** `problems/ode.py` (new `lib/diff.py`)

Only first-order temporal derivatives are currently computed. PDE problems commonly require:

- Second-order derivatives: `d²u/dx²` (heat equation, wave equation)
- Mixed partials: `d²u/dxdy`
- Laplacian: `∇²u`

A composable differential operator utility (e.g., `grad(u, x)`, `laplacian(u, coords)`,
`divergence(F, coords)`) would make PDE constraint implementations concise and reusable without
re-implementing autograd boilerplate in each constraint class.

#### PDE4 — Multi-dimensional collocation sampling strategies

**Files:** `core/dataset.py`, `catalog/*.py`

`gen_coll()` currently produces shape `(M, 1)` tensors. Multi-dimensional PDE problems need
`(M, d)` collocation points, potentially over complex geometries.

The collocation strategy should be decoupled into a `CollocationSampler` protocol with
implementations:

```python
class CollocationSampler(Protocol):
    def sample(self, n: int, domain: Domain) -> Tensor: ...

class UniformGridSampler(CollocationSampler): ...
class RandomUniformSampler(CollocationSampler): ...
class LatinHypercubeSampler(CollocationSampler): ...
class AdaptiveResidueSampler(CollocationSampler): ...   # residual-based refinement
```

Adaptive sampling — concentrating collocation points where the PDE residual is large — is a known
technique for improving PINN accuracy on problems with sharp gradients and would be a
differentiating feature.

#### PDE6 — Spatial input encodings for `Field`

**File:** `core/nn.py`

`MLPConfig.encode` supports a custom encoding callback but ships no built-in spatial encodings.
For PDE problems with high-frequency solutions (wave equations, turbulence), plain coordinate
inputs cause spectral bias. Standard encodings that should be provided out of the box:

- **Random Fourier features**: `[sin(Bx), cos(Bx)]` where `B` is sampled at initialization
- **Positional encoding**: sinusoidal encoding as in the original transformer paper
- **Hash grid encoding**: multi-resolution hash (as in Instant-NGP)

#### PDE7 — Configurable loss criterion

**File:** `problems/ode.py`

`ODEInverseProblem` hard-codes `nn.MSELoss()`. For multi-scale PDEs where residuals span several
orders of magnitude, MSE is dominated by the largest component. The criterion should be
configurable:

```python
class ODEInverseProblem(Problem):
    def __init__(self, ..., criterion: nn.Module = nn.MSELoss()): ...
```

Useful alternatives include Huber loss, relative L2 loss, and component-weighted MSE.

#### PDE8 — Scoped constraints for coupled PDE systems

**File:** `core/problem.py`

`Problem` currently applies all constraints to all fields uniformly. Coupled PDE systems — for
example, Navier-Stokes where velocity and pressure satisfy different equations — need constraints
scoped to specific subsets of fields. The continuity equation applies only to pressure; the
momentum equation applies only to velocity.

One design: `Constraint` accepts an optional `fields_scope: list[str]` argument, and `Problem`
filters the `FieldsRegistry` before passing it to each constraint. This preserves backward
compatibility while enabling coupled systems.

---

### 2.3 Recommended Implementation Order

For a contributor picking up this work, the recommended order is:

1. **D1, D2, D7** — Fast, high-impact DX fixes. Small effort, eliminates the most common failure modes.
2. **D5, D6** — Correctness fixes for validation and data loading.
3. **D3, D4, D8** — Construction-time validation and documentation.
4. **PDE1 + PDE2 + PDE5** — Must be tackled together as a single coherent change.
5. **PDE3, PDE4** — Build on top of the domain/BC abstraction.
6. **PDE6, PDE7, PDE8** — Quality-of-life for PDE users.

---

## Summary

AnyPINN occupies a specific niche that is genuinely unoccupied by existing libraries: a
**PyTorch-native, training-engine-agnostic library where ODE inverse problems are expressed as
composable `nn.Module` objects**, with a bootstrapper that eliminates cold-start friction and a
typed configuration system that makes misconfiguration a compile-time error rather than a runtime
one.

The library's justification is strongest for researchers working on parameter recovery in
dynamical systems — epidemiological models, mechanical oscillators, predator-prey dynamics — who
want to define physics in PyTorch terms and bring their own training infrastructure. The PDE
expansion track would extend this justification to a substantially larger problem class, but it
requires the domain abstraction refactor (PDE1) as its foundation.
