# Inverse Diffusivity

```bash
anypinn create my-project --template inverse-diffusivity
```

Recovers a space-dependent diffusivity D(x) represented as a neural network `Field`, rather than a
single scalar parameter.

## Problem

$$
\frac{\partial u}{\partial t} = \nabla \cdot \bigl(D(x)\,\nabla u\bigr)
$$

where D(x) is unknown and learned as a neural network.

## Features Demonstrated

- Function-valued parameter recovery (D(x) as a `Field`)
- Composable differential operators from `anypinn.lib.diff`
- PDE inverse problem with spatially varying coefficients

## Results

![Inverse Diffusivity results](../examples/inverse_diffusivity/results/inverse-diffusivity.png)
