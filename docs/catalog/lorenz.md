# Lorenz System

```bash
anypinn create my-project --template lorenz
```

Chaotic 3-field ODE. Recovers $\sigma$, $\rho$, and $\beta$ simultaneously from trajectory
observations.

## Background

The Lorenz system, derived by Edward Lorenz (1963) as a simplified model of atmospheric
convection, is one of the earliest and most studied examples of deterministic chaos. For the
classical parameter values ($\sigma = 10$, $\rho = 28$, $\beta = 8/3$) the system exhibits
extreme sensitivity to initial conditions (the "butterfly effect"). Trajectories settle
onto a fractal **strange attractor** that never repeats. Recovering all three parameters
simultaneously from a noisy, chaotic trajectory tests the robustness of the inverse solver;
Huber loss is used to mitigate the effect of outlier residuals.

## Governing Equations

$$
\begin{cases}
\dot{x} = \sigma\,(y - x) \\[6pt]
\dot{y} = x\,(\rho - z) - y \\[6pt]
\dot{z} = x\,y - \beta\,z
\end{cases}
$$

where:

- $x(t)$: proportional to convective circulation intensity
- $y(t)$: proportional to temperature difference between ascending and descending currents
- $z(t)$: proportional to deviation from linear vertical temperature profile
- $\sigma$: Prandtl number (**to recover**)
- $\rho$: normalized Rayleigh number (**to recover**)
- $\beta$: geometric factor (**to recover**)

## Default Configuration

The generated template uses the following values.

**Parameters to recover:**

| Symbol | Code constant | True value |
|--------|---------------|------------|
| $\sigma$ | `TRUE_SIGMA` | $10.0$ |
| $\rho$ | `TRUE_RHO` | $28.0$ |
| $\beta$ | `TRUE_BETA` | $8/3 \approx 2.667$ |

**Initial conditions:** $x(0) = -8, \quad y(0) = 7, \quad z(0) = 27$

**Domain:** $t \in [0, 3]$

**Scaling:** state variables are divided by $S = 20$ in the training ODE.

## Features Demonstrated

- Multi-parameter recovery (3 simultaneous parameters)
- Huber loss via `PINNHyperparameters.criterion` for robustness to chaotic trajectories
- 3-field system

## Results

![Lorenz results](../examples/lorenz/results/lorenz.png)
