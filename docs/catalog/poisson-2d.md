# Poisson 2D

```bash
anypinn create my-project --template poisson-2d
```

2D elliptic PDE forward problem. Solves for the solution field given a known source term and
boundary conditions.

## Background

The Poisson equation is one of the most fundamental elliptic PDEs in mathematical physics. It
arises in electrostatics (computing the electric potential from a charge distribution),
gravitational theory, and steady-state heat conduction. As a time-independent equation it
describes equilibrium states, making it the simplest setting for demonstrating PDE residual
and boundary condition constraints. The template uses a manufactured solution with an
analytically known source term, so the forward solve can be verified exactly.

## Governing Equations

$$
-\nabla^2 u = f(x, y)
$$

with source term:

$$
f(x, y) = -2\pi^2 \sin(\pi x)\,\sin(\pi y)
$$

where:

- $u(x, y)$: solution field
- $\nabla^2 = \dfrac{\partial^2}{\partial x^2} + \dfrac{\partial^2}{\partial y^2}$: 2D Laplacian
- $f(x, y)$: prescribed source (code: `source_fn`)

## Default Configuration

The generated template uses the following values.

**Boundary conditions:** $u = 0$ on $\partial\Omega$ (homogeneous Dirichlet on all four edges).

**Domain:** $(x, y) \in [0, 1]^2$

In the **inverse** variant (`--direction inverse`), a scaling parameter $k$ is introduced as
$-k\,\nabla^2 u = f$ with true value $k = 1$ (code: `TRUE_K`).

## Features Demonstrated

- `PDEResidualConstraint` for interior residual enforcement
- `DirichletBCConstraint` for boundary condition enforcement
- Forward problem (no parameter recovery)

## Results

![Poisson 2D results](../examples/poisson_2d/results/poisson-2d.png)
