# Forward vs Inverse Problems

AnyPINN supports both forward and inverse problems. This guide explains the
difference and how the library handles each.

---

## What's the difference?

**Forward problem:** All parameters are known. The PINN approximates the
solution of the differential equation.

$$
\text{Given } \frac{dy}{dt} = f(t, y, \theta) \text{ with known } \theta,
\text{ find } y(t).
$$

**Inverse problem:** Some parameters are unknown. The PINN simultaneously
approximates the solution *and* recovers the parameters from partial
observations.

$$
\text{Given } \frac{dy}{dt} = f(t, y, \theta) \text{ and observations } y_i^{\text{obs}},
\text{ find } y(t) \text{ and } \theta.
$$

---

## Inverse problems in AnyPINN

This is AnyPINN's primary use case. The key abstraction is the
`Argument` / `Parameter` split:

```python
# Everything the equation needs is in args
props = ODEProperties(
    ode=my_ode,
    y0=torch.tensor([...]),
    args={
        "gamma": Argument(0.1),  # Known, fixed value
    },
)

# Unknown quantities go in params
params = ParamsRegistry({
    "beta": Parameter(config=hp.params_config),  # Unknown, learned
})
```

The callable accesses both the same way:

```python
def my_ode(x, y, args):
    beta = args["beta"](x)   # Works for both Argument and Parameter
    gamma = args["gamma"](x)
    ...
```

The `ODEInverseProblem` class composes three constraints:

1. **Residual loss**: enforces the differential equation on collocation points
2. **Initial condition loss**: enforces the initial state
3. **Data loss**: fits the predicted fields to observations

All three are necessary for a well-posed inverse problem.

---

## Forward problems in AnyPINN

For forward problems, there are no unknown parameters and typically no observed
data to fit against. The loss has only two terms:

1. **Residual loss**: enforces the differential equation
2. **Initial/boundary condition loss**: enforces constraints

AnyPINN supports forward problems through two paths:

### ODE forward problems

Use `ODEInverseProblem` with no parameters and no data constraint:

```python
props = ODEProperties(
    ode=my_ode,
    y0=torch.tensor([...]),
    args={
        "beta": Argument(0.3),
        "gamma": Argument(0.1),
    },
)

problem = ODEInverseProblem(
    props=props,
    hp=hp,
    fields=fields,
    params=ParamsRegistry({}),  # No learnable parameters
)
```

### PDE forward problems

Use the PDE constraint classes directly for more control over boundary
conditions:

```python
from anypinn.problems import (
    PDEResidualConstraint,
    DirichletBCConstraint,
    BoundaryCondition,
)
```

See [PDE Forward Problems](pde-forward-problems.md) for a full walkthrough.

---

## When to use which

| Scenario | Type | Example |
| -------- | ---- | ------- |
| Recover rate constants from epidemic data | Inverse | SIR model |
| Solve a heat equation with known conductivity | Forward | Heat 1D |
| Estimate diffusivity from temperature measurements | Inverse | Inverse Diffusivity |
| Compute the steady state of a reaction-diffusion system | Forward | Allen-Cahn |

Most of the built-in catalog templates are inverse problems. The PDE templates
(Poisson 2D, Allen-Cahn) demonstrate forward problems.
