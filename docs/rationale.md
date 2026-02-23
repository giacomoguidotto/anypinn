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

| Library        | Coupling mechanism                                |
| -------------- | ------------------------------------------------- |
| NeuroDiffEq    | `solve_ivp`-style API owns training               |
| DeepXDE        | `Model.train()` is not optional                   |
| IDRLNet        | Computational graph node system owns training     |
| NVIDIA Modulus | Full platform: owns Trainer, distributed, logging |
| PINA           | Own `Trainer` wrapping Lightning                  |

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

PyTorch Lightning is available as an _optional_ layer (`anypinn.lightning`) for users who want
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

### 2.1 PDE Maturity Track

#### PDE4 - Multi-dimensional collocation sampling

Current collocation generation needs a pluggable strategy layer for practical PDE quality:

- uniform/random baselines,
- Latin hypercube coverage,
- adaptive residual-driven refinement.

#### PDE6 - Built-in spatial encodings

`MLPConfig.encode` is extensible but ships no built-in PDE-focused encodings.
Add first-class options for random Fourier features and positional encodings (with hash-grid as an
optional advanced path).

#### PDE7 - Configurable loss criterion

`ODEInverseProblem` still hard-codes `nn.MSELoss()`.
Expose criterion selection to support multi-scale residuals (Huber, weighted MSE, relative losses).

#### PDE8 - Scoped constraints for coupled systems

Coupled PDE systems need constraints operating on field subsets (e.g. continuity vs momentum
components). Add explicit constraint scoping instead of forcing per-constraint manual filtering.

### 2.2 ODE Ergonomics Track

#### ODE1 - Native second-order ODE path

Second-order systems are currently modeled through first-order state augmentation.
Add native second-order abstractions:

- second-order residual constraint,
- callable protocol for `d2y/dx2`,
- initial derivative condition support.

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
requires more work to be done.
