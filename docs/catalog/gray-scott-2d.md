# Gray-Scott 2D

```bash
anypinn create my-project --template gray-scott-2d
```

Coupled two-field reaction-diffusion PDE. Recovers diffusion rates and reaction parameters from
pattern snapshots.

## Background

The Gray-Scott model describes two reacting and diffusing chemical species in a continuously
stirred tank reactor: a substrate $u$ is fed at rate $F$ and consumed by an autocatalytic reaction
$u + 2v \to 3v$, while the product $v$ decays at rate $F + k$. Depending on the feed rate $F$
and kill rate $k$, the system produces a rich variety of Turing-type patterns including spots, stripes,
labyrinthine structures, and self-replicating pulses. It is widely studied as a prototype for
self-organizing pattern formation in chemistry and developmental biology.

## Governing Equations

$$
\begin{cases}
\dfrac{\partial u}{\partial t} = D_u\,\nabla^2 u - u\,v^2 + F\,(1 - u) \\[8pt]
\dfrac{\partial v}{\partial t} = D_v\,\nabla^2 v + u\,v^2 - (F + k)\,v
\end{cases}
$$

where:

- $u(\mathbf{x}, t)$: substrate concentration
- $v(\mathbf{x}, t)$: product concentration
- $D_u, D_v$: diffusion coefficients (**to recover**)
- $F$: feed rate (**to recover**)
- $k$: kill rate (**to recover**)
- $\nabla^2 = \dfrac{\partial^2}{\partial x^2} + \dfrac{\partial^2}{\partial y^2}$: 2D Laplacian

## Default Configuration

The generated template uses the following values.

**Parameters to recover:**

| Symbol | Code constant | True value |
|--------|---------------|------------|
| $D_u$ | `TRUE_DU` | $5 \times 10^{-3}$ |
| $D_v$ | `TRUE_DV` | $2.5 \times 10^{-3}$ |
| $F$ | `TRUE_F` | $0.04$ |
| $k$ | `TRUE_K` | $0.06$ |

**Initial conditions:** $u = 1,\; v = 0$ everywhere, except a central square
$[0.4, 0.6]^2$ where $u = 0.5,\; v = 0.25$ (seeding the pattern).

**Domain:** $(x, y) \in [0, 1]^2, \quad t \in [0, 200]$

**Boundary conditions:** Neumann (zero-flux) on all edges.

## Features Demonstrated

- 2D PDE with coupled fields
- `PDEResidualConstraint` with field-subset scoping
- Multi-parameter recovery (diffusion rates and reaction parameters)

## Results

![Gray-Scott 2D results](../examples/gray_scott_2d/results/gray-scott-2d.png)
