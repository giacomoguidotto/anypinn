# Van der Pol Oscillator

```bash
anypinn create my-project --template van-der-pol
```

Second-order nonlinear oscillator. Recovers nonlinearity parameter μ.

## Problem

$$
\ddot{x} - \mu(1 - x^2)\dot{x} + x = 0
$$

## Features Demonstrated

- Second-order ODE support via `ODEProperties.order`
- Nonlinear dynamics with limit cycles
- Scalar `Parameter` recovery

## Results

![Van der Pol results](../examples/van_der_pol/results/van-der-pol.png)
