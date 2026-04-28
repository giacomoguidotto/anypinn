# Damped Oscillator

```bash
anypinn create my-project --template damped-oscillator
```

Harmonic oscillator with damping. Recovers damping ratio ζ from displacement observations.

## Problem

Second-order ODE with learnable damping ratio ζ:

$$
\ddot{x} + 2\zeta\omega_0 \dot{x} + \omega_0^2 x = 0
$$

## Features Demonstrated

- Second-order ODE support via `ODEProperties.order`
- Native higher-order initial condition enforcement (`ODEProperties.dy0`)
- Scalar `Parameter` recovery

## Results

![Damped Oscillator results](../examples/damped_oscillator/results/damped-oscillator.png)
