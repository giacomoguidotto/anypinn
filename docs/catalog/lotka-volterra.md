# Lotka-Volterra

```bash
anypinn create my-project --template lotka-volterra
```

Predator-prey dynamics. Recovers predation rate $\beta$ from population observations.

## Background

The Lotka-Volterra equations, formulated independently by Alfred Lotka (1925) and Vito Volterra
(1926), are the classical model of predator-prey interaction in ecology. The system exhibits
characteristic oscillatory cycles: prey abundance fuels predator growth, which in turn suppresses
prey, leading to predator decline and eventual prey recovery. The four rate constants
($\alpha, \beta, \delta, \gamma$) control the amplitude and period of these cycles. The model is
a cornerstone of mathematical biology and population dynamics.

## Governing Equations

$$
\begin{cases}
\dfrac{dx}{dt} = \alpha\, x - \beta\, x\, y \\[8pt]
\dfrac{dy}{dt} = \delta\, x\, y - \gamma\, y
\end{cases}
$$

where:

- $x(t)$: prey population
- $y(t)$: predator population
- $\alpha$: prey birth rate (known)
- $\beta$: predation rate (**to recover**)
- $\delta$: predator growth rate per prey consumed (known)
- $\gamma$: predator death rate (known)

## Default Configuration

The generated template uses the following values.

**Parameters to recover:**

| Symbol | Code constant | True value |
|--------|---------------|------------|
| $\beta$ | `TRUE_BETA` | $0.02$ |

**Known constants:**

| Symbol | Code constant | Value |
|--------|---------------|-------|
| $\alpha$ | `TRUE_ALPHA` | $0.5$ |
| $\delta$ | `TRUE_DELTA` | $0.01$ |
| $\gamma$ | `TRUE_GAMMA` | $0.5$ |

**Initial conditions:** $x(0) = 40, \quad y(0) = 9$

**Domain:** $t \in [0, 50]$

**Scaling:** populations are divided by $P = 100$ in the training ODE.

## Features Demonstrated

- `FourierEncoding` for capturing oscillatory solutions
- Scalar `Parameter` recovery
- `ValidationRegistry` for ground-truth comparison

## Results

![Lotka-Volterra results](../examples/lotka_volterra/results/lotka-volterra.png)
