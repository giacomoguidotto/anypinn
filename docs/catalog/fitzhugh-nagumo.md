# FitzHugh-Nagumo

```bash
anypinn create my-project --template fitzhugh-nagumo
```

Two-field nonlinear neuron model. Recovers timescale ε from a partially observed voltage trace.

## Problem

Two fields (v, w), one learnable parameter (ε):

$$
\dot{v} = v - \frac{v^3}{3} - w + I_{\text{ext}}, \qquad
\dot{w} = \varepsilon(v + a - bw)
$$

## Features Demonstrated

- Partial observability (only voltage is observed)
- Scalar `Parameter` recovery
- Neural excitation dynamics

## Results

![FitzHugh-Nagumo results](../examples/fitzhugh_nagumo/results/fitzhugh-nagumo.png)
