# Lotka-Volterra

```bash
anypinn create my-project --template lotka-volterra
```

Predator-prey dynamics. Recovers predation rate β from population observations.

## Problem

Two fields (prey, predator), one learnable parameter (β):

$$
\frac{dx}{dt} = \alpha x - \beta x y, \qquad
\frac{dy}{dt} = \delta x y - \gamma y
$$

## Features Demonstrated

- `FourierEncoding` for capturing oscillatory solutions
- Scalar `Parameter` recovery
- `ValidationRegistry` for ground-truth comparison

## Results

![Lotka-Volterra results](../examples/lotka_volterra/results/lotka-volterra.png)
