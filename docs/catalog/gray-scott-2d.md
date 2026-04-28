# Gray-Scott 2D

```bash
anypinn create my-project --template gray-scott-2d
```

Coupled two-field reaction-diffusion PDE. Recovers diffusion rates and reaction parameters from
pattern snapshots.

## Problem

$$
\frac{\partial u}{\partial t} = D_u \nabla^2 u - uv^2 + F(1-u), \qquad
\frac{\partial v}{\partial t} = D_v \nabla^2 v + uv^2 - (F+k)v
$$

## Features Demonstrated

- 2D PDE with coupled fields
- `PDEResidualConstraint` with field-subset scoping
- Multi-parameter recovery (diffusion rates and reaction parameters)

## Results

![Gray-Scott 2D results](../examples/gray_scott_2d/results/gray-scott-2d.png)
