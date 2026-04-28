# Allen-Cahn

```bash
anypinn create my-project --template allen-cahn
```

Stiff reaction-diffusion PDE with sharp interfaces. Forward problem demonstrating adaptive sampling
and periodic boundary conditions.

## Problem

$$
\frac{\partial u}{\partial t} = \varepsilon^2 \frac{\partial^2 u}{\partial x^2} + u - u^3
$$

## Features Demonstrated

- `AdaptiveSampler` for resolving sharp interfaces
- Periodic boundary conditions
- Stiff PDE dynamics (small ε)
- Forward problem (no parameter recovery)

## Results

![Allen-Cahn results](../examples/allen_cahn/results/allen-cahn.png)
