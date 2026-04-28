# Heat Equation 1D

```bash
anypinn create my-project --template heat-1d
```

1D parabolic PDE inverse problem. Recovers thermal diffusivity α from sparse temperature
measurements.

## Problem

$$
\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}
$$

## Features Demonstrated

- PDE inverse problem (scalar parameter recovery)
- Sparse observation data
- Parabolic PDE handling

## Results

![Heat 1D results](../examples/heat_1d/results/heat-1d.png)
