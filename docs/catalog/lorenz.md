# Lorenz System

```bash
anypinn create my-project --template lorenz
```

Chaotic 3-field ODE. Recovers σ, ρ, and β simultaneously from trajectory observations.

## Problem

Three fields (x, y, z), three learnable parameters:

$$
\dot{x} = \sigma(y - x), \qquad
\dot{y} = x(\rho - z) - y, \qquad
\dot{z} = xy - \beta z
$$

## Features Demonstrated

- Multi-parameter recovery (3 simultaneous parameters)
- Huber loss via `PINNHyperparameters.criterion` for robustness to chaotic trajectories
- 3-field system

## Results

![Lorenz results](../examples/lorenz/results/lorenz.png)
