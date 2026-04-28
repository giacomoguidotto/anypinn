# Poisson 2D

```bash
anypinn create my-project --template poisson-2d
```

2D elliptic PDE forward problem. Solves for the solution field given a known source term and
boundary conditions.

## Problem

$$
-\nabla^2 u = f(x, y)
$$

with Dirichlet boundary conditions on the domain boundary.

## Features Demonstrated

- `PDEResidualConstraint` for interior residual enforcement
- `DirichletBCConstraint` for boundary condition enforcement
- Forward problem (no parameter recovery)

## Results

![Poisson 2D results](../examples/poisson_2d/results/poisson-2d.png)
