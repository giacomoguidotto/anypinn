# Heat Equation 1D

```bash
anypinn create my-project --template heat-1d
```

1D parabolic PDE inverse problem. Recovers thermal diffusivity $\alpha$ from sparse temperature
measurements.

## Background

The heat equation describes thermal diffusion in a conducting medium according to Fourier's law.
It is the prototypical parabolic PDE: an initial temperature profile smooths over time at a rate
controlled by the thermal diffusivity $\alpha$. The template uses a separable exact solution
(exponential decay times a spatial mode) so that the recovered $\alpha$ can be verified against
the known analytical value. This makes it an ideal first example for PDE-based inverse problems.

## Governing Equations

$$
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}
$$

where:

- $u(x, t)$: temperature field
- $\alpha$: thermal diffusivity (**to recover**)

Exact solution used for data generation:

$$
u(x, t) = e^{-\alpha \pi^2 t}\,\sin(\pi x)
$$

## Default Configuration

The generated template uses the following values.

**Parameters to recover:**

| Symbol | Code constant | True value |
|--------|---------------|------------|
| $\alpha$ | `TRUE_ALPHA` | $0.1$ |

**Initial condition:** $u(x, 0) = \sin(\pi x)$

**Boundary conditions:** $u(0, t) = u(1, t) = 0$ (homogeneous Dirichlet).

**Domain:** $x \in [0, 1], \quad t \in [0, 1]$

## Features Demonstrated

- PDE inverse problem (scalar parameter recovery)
- Sparse observation data
- Parabolic PDE handling

## Results

![Heat 1D results](../examples/heat_1d/results/heat-1d.png)
