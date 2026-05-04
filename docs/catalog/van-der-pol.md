# Van der Pol Oscillator

```bash
anypinn create my-project --template van-der-pol
```

Second-order nonlinear oscillator. Recovers nonlinearity parameter $\mu$.

## Background

The Van der Pol oscillator, proposed by Balthasar van der Pol (1926) to model oscillations in
vacuum tube circuits, is a canonical example of a self-excited oscillator. When $\mu > 0$ the
system has negative damping for small amplitudes (energy injection) and positive damping for large
amplitudes (energy dissipation), producing a stable **limit cycle**, a self-sustaining periodic
orbit whose shape and amplitude are independent of initial conditions. For small $\mu$ the
waveform is nearly sinusoidal; for large $\mu$ it becomes a **relaxation oscillation** with
alternating slow drifts and rapid jumps.

## Governing Equations

$$
\ddot{x} - \mu\,(1 - x^2)\,\dot{x} + x = 0
$$

Equivalently, with $v = \dot{x}$:

$$
\begin{cases}
\dfrac{dx}{dt} = v \\[8pt]
\dfrac{dv}{dt} = \mu\,(1 - x^2)\,v - x
\end{cases}
$$

where:

- $x(t)$: displacement (state variable)
- $v(t)$: velocity
- $\mu$: nonlinearity / damping strength (**to recover**)

## Default Configuration

The generated template uses the following values.

**Parameters to recover:**

| Symbol | Code constant | True value |
|--------|---------------|------------|
| $\mu$ | `TRUE_MU` | $1.0$ |

**Initial conditions:** $x(0) = 2.0, \quad \dot{x}(0) = 0.0$

**Domain:** $t \in [0, 20]$ s

## Features Demonstrated

- Second-order ODE support via `ODEProperties.order`
- Nonlinear dynamics with limit cycles
- Scalar `Parameter` recovery

## Results

![Van der Pol results](../examples/van_der_pol/results/van-der-pol.png)
