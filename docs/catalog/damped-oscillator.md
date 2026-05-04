# Damped Oscillator

```bash
anypinn create my-project --template damped-oscillator
```

Harmonic oscillator with damping. Recovers damping ratio $\zeta$ from displacement observations.

## Background

The damped harmonic oscillator is a fundamental model in mechanics, electrical engineering, and
control theory. It describes a system subject to a restoring force proportional to displacement
and a dissipative force proportional to velocity. The damping ratio $\zeta$ classifies the
response into three regimes: **underdamped** ($\zeta < 1$, oscillatory decay),
**critically damped** ($\zeta = 1$, fastest non-oscillatory return), and **overdamped**
($\zeta > 1$, sluggish return). The template uses an underdamped configuration
($\zeta = 0.15$).

## Governing Equations

Second-order ODE with learnable damping ratio $\zeta$:

$$
\ddot{x} + 2\zeta\omega_0\,\dot{x} + \omega_0^2\,x = 0
$$

Equivalently, as a first-order system with velocity $v = \dot{x}$:

$$
\begin{cases}
\dfrac{dx}{dt} = v \\[8pt]
\dfrac{dv}{dt} = -2\zeta\omega_0\,v - \omega_0^2\,x
\end{cases}
$$

where:

- $x(t)$: displacement
- $v(t)$: velocity
- $\omega_0$: natural frequency (known)
- $\zeta$: damping ratio (**to recover**)

## Default Configuration

The generated template uses the following values.

**Parameters to recover:**

| Symbol | Code constant | True value |
|--------|---------------|------------|
| $\zeta$ | `TRUE_ZETA` | $0.15$ |

**Known constants:**

| Symbol | Code constant | Value |
|--------|---------------|-------|
| $\omega_0$ | `TRUE_OMEGA0` | $2\pi \approx 6.283$ |

**Initial conditions:** $x(0) = 1.0, \quad \dot{x}(0) = 0.0$

**Domain:** $t \in [0, 5]$ s

## Features Demonstrated

- Second-order ODE support via `ODEProperties.order`
- Native higher-order initial condition enforcement (`ODEProperties.dy0`)
- Scalar `Parameter` recovery

## Results

![Damped Oscillator results](../examples/damped_oscillator/results/damped-oscillator.png)
