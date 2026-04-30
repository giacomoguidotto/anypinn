# PDE Forward Problems

This guide shows how to set up a PDE problem with boundary conditions,
multi-dimensional domains, and PDE residuals using AnyPINN's constraint classes.

---

## Domain definition

PDEs operate on multi-dimensional domains. Define the domain with bounds for
each dimension:

```python
from anypinn.core import Domain

# 2D spatial domain: x in [0, 1], y in [0, 1]
domain = Domain(bounds=[(0.0, 1.0), (0.0, 1.0)])

# 1D spatial + time: x in [0, 1], t in [0, 2]
domain = Domain(bounds=[(0.0, 1.0), (0.0, 2.0)])
```

---

## Boundary conditions

AnyPINN provides three boundary condition types:

### Dirichlet: prescribe the value

$$u(x_{\text{bc}}) = g(x_{\text{bc}})$$

```python
from anypinn.problems import BoundaryCondition, DirichletBCConstraint

bc_left = BoundaryCondition(
    sampler=lambda n: torch.stack([
        torch.zeros(n),           # x = 0
        torch.rand(n) * T_max,   # t uniform in [0, T]
    ], dim=-1),
    value=lambda coords: torch.zeros(coords.shape[0], 1),
    n_pts=50,
)

dirichlet = DirichletBCConstraint(
    bc=bc_left,
    field=fields["u"],
    weight=10.0,
)
```

### Neumann: prescribe the derivative

$$\frac{\partial u}{\partial n}(x_{\text{bc}}) = h(x_{\text{bc}})$$

```python
from anypinn.problems import NeumannBCConstraint

neumann = NeumannBCConstraint(
    bc=bc_right,
    field=fields["u"],
    component=0,          # Which spatial dimension for the normal derivative
    weight=10.0,
)
```

### Periodic: match values at opposite boundaries

```python
from anypinn.problems import PeriodicBCConstraint

periodic = PeriodicBCConstraint(
    field=fields["u"],
    sampler_left=lambda n: ...,
    sampler_right=lambda n: ...,
    n_pts=50,
    weight=10.0,
)
```

---

## PDE residual

Define the PDE residual — the quantity that should be zero at the solution.
AnyPINN computes spatial derivatives using automatic differentiation:

```python
from anypinn.problems import PDEResidualConstraint
from anypinn.lib.diff import partial

def heat_residual(x, fields, params):
    u = fields["u"]

    # x has shape (n_pts, 2): columns are [spatial_x, time_t]
    u_val = u(x)
    du_dt = partial(u_val, x, dim=1)      # ∂u/∂t
    du_dx = partial(u_val, x, dim=0)      # ∂u/∂x
    d2u_dx2 = partial(du_dx, x, dim=0)    # ∂²u/∂x²

    alpha = params["alpha"](x)
    return du_dt - alpha * d2u_dx2         # Should be zero

residual_constraint = PDEResidualConstraint(
    residual_fn=heat_residual,
    fields=fields,
    params=params,
    weight=1.0,
)
```

---

## Assembling the problem

For PDE problems, compose constraints manually using the base `Problem` class
instead of `ODEInverseProblem`:

```python
from anypinn.core import Problem

problem = Problem(
    constraints=[
        residual_constraint,
        dirichlet_left,
        dirichlet_right,
    ],
    fields=fields,
    params=params,
    hp=hp,
)
```

The `Problem` sums the weighted constraint losses and manages the shared
field and parameter registries.

---

## Collocation sampling for PDEs

PDEs typically need more collocation points than ODEs because the domain is
higher-dimensional. Configure the sampler:

```python
from anypinn.core import GenerationConfig

hp = PINNHyperparameters(
    training_data=GenerationConfig(
        collocations=10000,    # 10k points for 2D
        batch_size=256,
    ),
    ...
)
```

For problems with sharp gradients or shocks (e.g. Burgers equation), use
adaptive collocation to concentrate points where the residual is largest:

```python
from anypinn.lightning.callbacks import AdaptiveCollocationCallback

trainer = pl.Trainer(
    callbacks=[
        AdaptiveCollocationCallback(resample_every=50),
    ],
)
```

---

## Catalog examples

Several catalog templates demonstrate PDE forward problems:

- **Poisson 2D** — elliptic PDE with Dirichlet boundaries
- **Allen-Cahn** — stiff reaction-diffusion
- **Heat 1D** — parabolic PDE with diffusivity recovery
- **Burgers 1D** — nonlinear PDE with adaptive collocation

Browse the [Catalog](../catalog/index.md) for complete implementations.
