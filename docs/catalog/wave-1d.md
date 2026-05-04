# Wave Equation 1D

```bash
anypinn create my-project --template wave-1d
```

1D hyperbolic PDE inverse problem. Recovers wave speed $c$ from sparse displacement measurements.

## Background

The wave equation governs the propagation of disturbances in elastic media such as vibrating strings,
acoustic pressure waves, and electromagnetic fields. As a hyperbolic PDE it preserves the shape
of initial disturbances (unlike the heat equation, which smooths them), transporting energy at
speed $c$ without dissipation. The template uses a standing-wave exact solution (product of
spatial and temporal modes) so that the recovered wave speed can be verified analytically.

## Governing Equations

$$
\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}
$$

where:

- $u(x, t)$: displacement field
- $c$: wave propagation speed (**to recover**)

Exact solution used for data generation:

$$
u(x, t) = \sin(\pi x)\,\cos(c\,\pi\,t)
$$

## Default Configuration

The generated template uses the following values.

**Parameters to recover:**

| Symbol | Code constant | True value |
|--------|---------------|------------|
| $c$ | `TRUE_C` | $1.0$ |

**Initial conditions:** $u(x, 0) = \sin(\pi x), \quad \dfrac{\partial u}{\partial t}(x, 0) = 0$

**Boundary conditions:** $u(0, t) = u(1, t) = 0$ (homogeneous Dirichlet).

**Domain:** $x \in [0, 1], \quad t \in [0, 1]$

## Features Demonstrated

- Hyperbolic PDE handling
- Scalar parameter recovery (wave speed)
- Sparse observation data

## Results

![Wave 1D results](../examples/wave_1d/results/wave-1d.png)
