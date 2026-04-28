# SEIR Epidemic Model

```bash
anypinn create my-project --template seir
```

Extended epidemic model with an exposed compartment E. Recovers transmission rate β from partially
observed data.

## Problem

Three fields (S, E, I), one learnable scalar parameter (β):

$$
\frac{dS}{dt} = -\beta \frac{SI}{N}, \qquad
\frac{dE}{dt} = \beta \frac{SI}{N} - \sigma E, \qquad
\frac{dI}{dt} = \sigma E - \delta I
$$

## Features Demonstrated

- Multi-field ODE system (3 fields)
- Scalar `Parameter` recovery
- `ValidationRegistry` for ground-truth comparison

## Results

![SEIR Inverse results](../examples/seir_inverse/results/seir-inverse.png)
