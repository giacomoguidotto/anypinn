---
title: "AnyPINN: A Modular, Training-Agnostic Python Library for Physics-Informed Neural Networks"
authors:
  - name: Giacomo Guidotto
    orcid: 0009-0006-2279-9126
    affiliation: 1
  - name: Damiano Pasetto
    orcid: 0000-0001-6892-9826
    affiliaion: 1
  - name: Caterina Millevoi
    orcid: 0000-0002-9743-4547
    affiliation: 2
affiliations:
  - name: Ca' Foscari University of Venice, Venice, IT
    index: 1
  - name: Università di Padova, Padova, IT
    index: 2
date: 27 February 2026
bibliography: paper.bib
---

# Summary

AnyPINN is a PyTorch-native library for expressing Physics-Informed Neural Networks (PINNs) [@raissi2019physics] as composable `nn.Module` objects. It supports both ODE and PDE problems, forward and inverse alike, with particular depth in the inverse setting where unknown parameters must be recovered from partial observations. The library is organised into four layers (`core`, `problems`, `lightning`, `catalog`) with a strict one-way dependency direction, so users can engage with exactly the layer that matches their expertise, from running a pre-built example through the bootstrap CLI to defining custom physics and training loops in plain PyTorch [@pytorch]. All hyperparameters are plain Python dataclasses: serializable, diffable, and programmatically constructible, which makes systematic ablation studies a matter of iterating over configuration objects rather than editing files. Training is not owned by the library: `Problem.training_loss()` returns a scalar tensor compatible with any optimizer or training infrastructure, including plain PyTorch and PyTorch Lightning [@lightning].

# Statement of Need

Physics-Informed Neural Networks encode differential equation residuals as loss terms, enabling mesh-free, automatic-differentiation-based solutions to both forward and inverse problems [@lagaris1998artificial; @raissi2019physics; @karniadakis2021physics; @cuomo2022scientific]. Forward problems seek the solution field given known equations and boundary or initial conditions; inverse problems additionally recover unknown parameters from partial observations such as transmission rates in epidemiology, damping coefficients in mechanics, interaction strengths in ecology.

Despite a growing ecosystem of PINN libraries [@lu2021deepxde; @chen2021neurodiffeq; @cuomo2022scientific], two pain points persist. First, every major library couples physics definition to its own training loop, preventing researchers from using the broader PyTorch ecosystem directly. Second, inverse problems remain second-class: unknown parameters are configuration flags rather than typed objects, and ground-truth tracking during training requires user-written callbacks.

AnyPINN addresses both gaps. The core abstraction, `Problem`, is a standard `nn.Module` whose `training_loss()` returns a scalar tensor, compatible with any optimizer, scheduler, or distributed strategy without library-specific wrappers. For inverse problems specifically, `Parameter` is a first-class type with the same `forward(x)` interface whether it wraps a fixed scalar or a time-varying MLP; switching from fixed to learnable requires changing one configuration line. `ValidationRegistry` binds ground-truth references to parameter names and logs MSE automatically at every training step. `ArgsRegistry` unifies fixed and learnable arguments so that the differential equation callable cannot distinguish between the two, enabling transparent composability.

The target audience ranges from researchers defining novel physics to engineers running parameter sweeps on pre-built models. The bootstrap CLI (`anypinn create`) scaffolds a complete project from a single interactive command, making the library accessible to users who have never written a PINN before. Researchers who need full control can bypass the CLI and the Lightning layer entirely, working directly with `anypinn.core` and plain PyTorch.

# State of the Field

The PINN software ecosystem spans more than ten libraries across two major frameworks [@cuomo2022scientific; @karniadakis2021physics]. \autoref{tab:comparison} compares the libraries most relevant to AnyPINN's scope across several dimensions.

| Library                            | Framework          | PDE support | Inverse (first-class)? | Scaffold CLI? | Engine-agnostic? |
| ---------------------------------- | ------------------ | ----------- | ---------------------- | ------------- | ---------------- |
| NeuroDiffEq [@chen2021neurodiffeq] | PyTorch            | 2D          | No                     | No            | No               |
| IDRLNet                            | PyTorch            | Yes         | No                     | No            | No               |
| NVIDIA Modulus [@modulus2023]      | PyTorch            | Yes (3D)    | No                     | No            | No               |
| PINA [@pina2024]                   | PyTorch            | Yes         | No                     | No            | No               |
| DeepXDE [@lu2021deepxde]           | TF / JAX / PyTorch | Yes         | No                     | No            | No               |
| TensorDiffEq                       | TensorFlow         | Yes         | No                     | No            | No               |
| SciANN                             | TensorFlow         | Yes         | No                     | No            | No               |
| PyDEns                             | TensorFlow         | Limited     | No                     | No            | No               |
| Elvet                              | TensorFlow         | Limited     | No                     | No            | No               |
| NVIDIA SimNet                      | TensorFlow         | Yes (3D)    | No                     | No            | No               |
| **AnyPINN**                        | **PyTorch**        | **Yes**     | **Yes**                | **Yes**       | **Yes**          |

Table: Comparison of AnyPINN with existing PINN libraries. "Inverse (first-class)" means the library provides typed parameter objects, unified fixed/learnable interfaces, and built-in ground-truth tracking, not merely the ability to mark a variable as trainable. \label{tab:comparison}

Three structural patterns in the existing ecosystem motivate AnyPINN as a separate library rather than a contribution to an existing one.

**Framework distribution.** More than half of the libraries in the comparison set are TensorFlow-based. PyTorch [@pytorch] has become the dominant framework for ML research, and its downstream ecosystem (PyTorch Lightning [@lightning], Hugging Face, `torch.compile`, Triton) has grown substantially. DeepXDE has added PyTorch and JAX backends, but was originally designed around TensorFlow idioms. A researcher already working in PyTorch adopting a TensorFlow-native PINN library inherits framework friction unrelated to the science.

**Training engine coupling.** Every major competitor couples physics definition to its own training loop: NeuroDiffEq exposes a `solve_ivp`-style API; DeepXDE requires calling `Model.train()`; IDRLNet uses a proprietary computational graph; NVIDIA Modulus ships its own distributed `Trainer`; PINA wraps Lightning with an additional abstraction layer. When the training ecosystem evolves with new optimizers, new precision formats, new distributed strategies, users of these libraries must wait for library-level support or write wrappers around internal abstractions. AnyPINN's `Problem` is a plain `nn.Module`; the library never owns the training loop.

**Inverse problem primitives.** All listed libraries can, in principle, solve inverse problems by marking variables as trainable. However, none ship `Parameter` as a typed object with a unified interface, a `ValidationRegistry` for ground-truth tracking, or an `ArgsRegistry` that makes fixed and learnable arguments indistinguishable to the equation callable. In libraries such as DeepXDE and NeuroDiffEq, the distinction between fixed and learnable parameters is visible at the problem-definition level, and ground-truth comparison requires user-written callbacks.

# Software Design

AnyPINN is split into four layers with strict one-way dependency (outer depends on inner, never the reverse).

**`anypinn.core`: pure PyTorch.**
The mathematical foundation defines what a PINN problem _is_ without any opinion about training. `Problem` (an `nn.Module`) aggregates `Constraint` instances, fields, and parameters and exposes `training_loss(batch, log)` returning a scalar tensor. `Field` is an MLP mapping input coordinates to state variables. `Parameter` inherits from `Argument` and exposes `forward(x)` whether it wraps a fixed scalar or a learnable MLP, making it transparent to the equation callable. `ArgsRegistry` is a typed dictionary of `Argument` instances; `InferredContext` extracts domain bounds from data and injects them into constraints automatically. Because `Problem` is a standard `nn.Module`, it participates transparently in the full PyTorch ecosystem: mixed precision, `torch.compile`, distributed data parallel, and gradient checkpointing require no library-level changes.

**`anypinn.problems`: ODE and PDE building blocks.**
For ODE inverse problems, `ResidualsConstraint` enforces $\|\dot{y} - f(t,y)\|^2$ via autograd with support for arbitrary-order ODEs through chained derivatives. `ICConstraint` enforces $\|y(t_0) - y_0\|^2$ and natively handles higher-order initial conditions. `DataConstraint` enforces $\|\hat{y} - y_{\text{obs}}\|^2$ against observed data. `ODEInverseProblem` composes all three with configurable loss weights and a selectable loss criterion (MSE, Huber, or L1). For PDE problems, `DirichletBCConstraint` and `NeumannBCConstraint` enforce boundary conditions on sampled boundary points, and `PDEResidualConstraint` enforces an arbitrary residual function over interior collocation points with field-subset scoping for coupled systems. The composable differential operators in `anypinn.lib.diff` (`grad`, `partial`, `mixed_partial`, `laplacian`, `divergence` and `hessian`) are available for use inside any constraint definition.

**`anypinn.lightning`: optional training wrapper.**
A thin PyTorch Lightning [@lightning] layer for users who want batteries-included training. `PINNModule` wraps any `Problem` as a `LightningModule`; `PINNDataModule` manages data loading and collocation sampling. Five collocation strategies are available via a configuration string: `"uniform"`, `"random"`, `"latin_hypercube"`, `"log_uniform_1d"`, and `"adaptive"` (residual-weighted, with configurable exploration ratio). Callbacks provide SMMA-based early stopping, data scaling, and prediction writing. The entire Lightning layer is optional: `anypinn.core` has no Lightning dependency.

**`anypinn.catalog`: drop-in problem modules.**
Pre-built ODE functions and data modules for SIR, SEIR, damped oscillator, and Lotka-Volterra systems. Catalog entries are runnable out of the box and serve as concrete starting points for the bootstrap CLI. The catalog is designed to be extended with additional dynamical systems as the library grows.

**Configuration as data.**
All hyperparameters such as MLP architecture, optimizer settings, collocation strategy, loss
weights, are expressed as keyword-only Python dataclasses. These objects are plain data:
they can be serialized, loaded from configuration files, diffed across runs, and
constructed programmatically in loops. This makes systematic ablation studies i.e., sweeping
over learning rates, collocation counts, or loss weights, a matter of iterating over
configuration objects in a script rather than editing files by hand.

**Input encodings.**
`FourierEncoding` and `RandomFourierFeatures` are available as `nn.Module` input encodings,
lifting low-frequency MLPs to high-frequency solutions without changes to the training loop.

The four-step workflow for a custom ODE inverse problem is:

```python
# 1. Define the ODE
def my_ode(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    k = args["k"](x)   # same interface for fixed or learnable k
    return -k * y

# 2. Configure hyperparameters
@dataclass(frozen=True, kw_only=True)
class MyHP(ODEHyperparameters):
    pde_weight: float = 1.0
    ic_weight: float = 10.0
    data_weight: float = 5.0

# 3. Build the problem
props = ODEProperties(ode=my_ode, args={"k": param}, y0=y0)
problem = ODEInverseProblem(ode_props=props, fields={"u": field},
                            params={"k": param}, hp=hp)

# 4. Train with Lightning or plain PyTorch
optimizer = torch.optim.Adam(problem.parameters(), lr=1e-3)
for batch in dataloader:
    optimizer.zero_grad()
    loss = problem.training_loss(batch, log=my_log_fn)
    loss.backward()
    optimizer.step()
```

The bootstrap CLI (`anypinn create`) generates a complete, immediately runnable four-file
project (`ode.py`, `config.py`, `train.py`, `pyproject.toml`) from a single interactive
command, lowering the barrier for users unfamiliar with PINN library APIs.

# Research Impact

AnyPINN ships six working examples spanning epidemiology (SIR with real Italian COVID-19
data, SEIR, reduced-SIR with hospitalization dynamics), mechanics (damped oscillator:
recovering damping ratio and natural frequency), and ecology (Lotka-Volterra predator-prey
dynamics), plus a minimal exponential decay script (~80 lines, core only, no Lightning).
Each example demonstrates a different dynamical system, scientific domain, and data type,
with generated result plots and CSV exports. Neural ODEs [@chen2018neural] offer a
complementary approach that replaces the ODE right-hand side with a neural network
entirely; AnyPINN instead keeps the equation explicit, which makes the same `Problem` and
`Constraint` abstractions applicable to both forward solutions and parameter recovery.
The PDE constraint layer (`DirichletBCConstraint`, `NeumannBCConstraint`,
`PDEResidualConstraint`) and the composable differential operators in `anypinn.lib.diff`
extend the library's applicability beyond ODE systems, providing a foundation for
spatiotemporal forward and inverse problems alike.

# AI Usage Disclosure

Claude (Anthropic) was used during the development of the AnyPINN codebase to assist with
code audit, bug identification, and targeted fixes. All architectural decisions, design
choices, and major implementations were authored by the human developer. Claude was not used
to generate the core library code or its scientific content.

# References
