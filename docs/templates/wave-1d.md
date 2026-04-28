# Wave Equation 1D

```bash
anypinn create my-project --template wave-1d
```

1D hyperbolic PDE inverse problem. Recovers wave speed c from sparse displacement measurements.

## Problem

$$
\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}
$$

## Features Demonstrated

- Hyperbolic PDE handling
- Scalar parameter recovery (wave speed)
- Sparse observation data

## Results

![Wave 1D results](../examples/wave_1d/results/wave-1d.png)
