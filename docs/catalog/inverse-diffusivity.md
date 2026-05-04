# Inverse Diffusivity

```bash
anypinn create my-project --template inverse-diffusivity
```

Recovers a space-dependent diffusivity $D(x)$ represented as a neural network `Field`, rather
than a single scalar parameter.

## Background

Recovering spatially varying material properties from indirect measurements is a central problem
in inverse theory, with applications in thermal conductivity imaging, permeability estimation in
porous media, and non-destructive material testing. Unlike the other PDE templates where the
unknown is a scalar, here the diffusivity $D(x)$ is a **function**. The template represents it
as a learnable neural network `Field`, demonstrating function-valued parameter recovery. The
expanded form of the divergence operator $\nabla \cdot (D\,\nabla u) = D\,u_{xx} + D'\,u_x$
shows how the spatial gradient of $D$ couples to the solution gradient.

## Governing Equations

$$
\frac{\partial u}{\partial t} = \nabla \cdot \bigl(D(x)\,\nabla u\bigr)
  = D(x)\,\frac{\partial^2 u}{\partial x^2} + D'(x)\,\frac{\partial u}{\partial x}
$$

where:

- $u(x, t)$: temperature / concentration field
- $D(x)$: spatially varying diffusivity (**to recover** as a neural network `Field`)

## Default Configuration

The generated template uses the following values.

**Function to recover:**

| Symbol | Code constant | True profile |
|--------|---------------|--------------|
| $D(x)$ | `TRUE_D_FN` | $0.1 + 0.05\,\sin(2\pi x)$ |

The diffusivity ranges from $0.05$ to $0.15$ over the spatial domain.

**Initial condition:** $u(x, 0) = \sin(\pi x)$

**Boundary conditions:** $u(0, t) = u(1, t) = 0$ (homogeneous Dirichlet).

**Domain:** $x \in [0, 1], \quad t \in [0, 1]$

## Features Demonstrated

- Function-valued parameter recovery ($D(x)$ as a `Field`)
- Composable differential operators from `anypinn.lib.diff`
- PDE inverse problem with spatially varying coefficients

## Results

![Inverse Diffusivity results](../examples/inverse_diffusivity/results/inverse-diffusivity.png)
