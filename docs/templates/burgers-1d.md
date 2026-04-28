# Burgers Equation 1D

```bash
anypinn create my-project --template burgers-1d
```

1D nonlinear PDE with shock formation. Recovers viscosity ν with adaptive collocation.

## Problem

$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu \frac{\partial^2 u}{\partial x^2}
$$

## Features Demonstrated

- Adaptive collocation via `AdaptiveCollocationCallback`
- Shock-forming nonlinear PDE
- Scalar parameter recovery (viscosity)

## Results

![Burgers 1D results](../examples/burgers_1d/results/burgers-1d.png)
