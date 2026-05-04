# Allen-Cahn

```bash
anypinn create my-project --template allen-cahn
```

Stiff reaction-diffusion PDE with sharp interfaces. Forward problem demonstrating adaptive
sampling and periodic boundary conditions.

## Background

The Allen-Cahn equation, introduced by Allen and Cahn (1979), models phase separation and
interface motion in binary alloys and other materials. The solution $u$ represents a
non-conserved order parameter that evolves under a double-well potential ($u - u^3$ term),
driving it toward the two stable phases $u \approx \pm 1$. The diffusion coefficient $\varepsilon$
controls the interface width; small $\varepsilon$ produces sharp transition layers between
phases, creating stiff dynamics that challenge standard PDE solvers and require adaptive
sampling to resolve accurately.

## Governing Equations

$$
\frac{\partial u}{\partial t} = \varepsilon\,\frac{\partial^2 u}{\partial x^2} + u - u^3
$$

where:

- $u(x, t)$: order parameter (phase field)
- $\varepsilon$: diffusion coefficient controlling interface width (known)

## Default Configuration

The generated template uses the following values.

**Known constants:**

| Symbol | Code constant | Value |
|--------|---------------|-------|
| $\varepsilon$ | `TRUE_EPSILON` | $0.01$ |

**Initial condition:** $u(x, 0) = -\tanh\!\left(\dfrac{x}{\sqrt{2\varepsilon}}\right)$, the
stationary kink profile centered at $x = 0$.

**Boundary conditions:** periodic ($u(-1, t) = u(1, t)$).

**Domain:** $x \in [-1, 1], \quad t \in [0, 1]$

## Features Demonstrated

- `AdaptiveSampler` for resolving sharp interfaces
- Periodic boundary conditions
- Stiff PDE dynamics (small $\varepsilon$)
- Forward problem (no parameter recovery)

## Results

![Allen-Cahn results](../examples/allen_cahn/results/allen-cahn.png)
