# SIR Epidemic Model

```bash
anypinn create my-project --template sir
```

Classic S→I→R compartmental model. Recovers transmission rate β from partially observed infected
counts.

## Problem

Two fields (S, I), one learnable scalar parameter (β), with known recovery rate δ and population N:

$$
\frac{dS}{dt} = -\beta \frac{SI}{N}, \qquad \frac{dI}{dt} = \beta \frac{SI}{N} - \delta I
$$

## Features Demonstrated

- Scalar `Parameter` recovery
- `ValidationRegistry` for ground-truth comparison
- `DataScaling` callback for population-scale normalization

## Results

![SIR Inverse results](../examples/sir_inverse/results/sir-inverse.png)
