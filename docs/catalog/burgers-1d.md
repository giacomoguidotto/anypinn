# Burgers Equation 1D

```bash
anypinn create my-project --template burgers-1d
```

1D nonlinear PDE with shock formation. Recovers viscosity $\nu$ with adaptive collocation.

## Background

The viscous Burgers equation combines nonlinear advection ($u\,\partial u / \partial x$) with
diffusion ($\nu\,\partial^2 u / \partial x^2$). It serves as a one-dimensional prototype for the
Navier-Stokes equations and is a standard benchmark for methods that must handle shock formation.
As the viscosity $\nu \to 0$ the solution develops steep gradients that evolve into shock-like
fronts, concentrating the PDE residual in narrow regions. This motivates the use of adaptive
collocation, which places more training points where the residual is largest.

## Governing Equations

$$
\frac{\partial u}{\partial t} + u\,\frac{\partial u}{\partial x} = \nu\,\frac{\partial^2 u}{\partial x^2}
$$

where:

- $u(x, t)$: velocity field
- $\nu$: kinematic viscosity (**to recover**)

## Default Configuration

The generated template uses the following values.

**Parameters to recover:**

| Symbol | Code constant | True value |
|--------|---------------|------------|
| $\nu$ | `TRUE_NU` | $0.01 / \pi \approx 0.00318$ |

**Initial condition:** $u(x, 0) = -\sin(\pi x)$

**Boundary conditions:** $u(-1, t) = u(1, t) = 0$ (homogeneous Dirichlet).

**Domain:** $x \in [-1, 1], \quad t \in [0, 1]$

## Features Demonstrated

- Adaptive collocation via `AdaptiveCollocationCallback`
- Shock-forming nonlinear PDE
- Scalar parameter recovery (viscosity)

## Results

![Burgers 1D results](../examples/burgers_1d/results/burgers-1d.png)
