# FitzHugh-Nagumo

```bash
anypinn create my-project --template fitzhugh-nagumo
```

Two-field nonlinear neuron model. Recovers timescale $\varepsilon$ and threshold parameter $a$
from a partially observed voltage trace.

## Background

The FitzHugh-Nagumo model, simplified by FitzHugh (1961) and Nagumo et al. (1962) from the
four-variable Hodgkin-Huxley equations, captures the essential dynamics of neural spike
generation with just two variables: a fast voltage-like variable $v$ and a slow recovery variable
$w$. The model exhibits **excitable** dynamics: small perturbations decay, but a sufficiently
large stimulus drives the system through a full action potential (spike) before returning to rest.
The timescale separation parameter $\varepsilon \ll 1$ governs how fast the recovery variable $w$
responds relative to the voltage $v$.

## Governing Equations

$$
\begin{cases}
\dfrac{dv}{dt} = v - \dfrac{v^3}{3} - w + I_{\text{ext}} \\[8pt]
\dfrac{dw}{dt} = \varepsilon\,(v + a - b\,w)
\end{cases}
$$

where:

- $v(t)$: membrane voltage (fast variable, **observed**)
- $w(t)$: recovery current (slow variable, **latent**)
- $\varepsilon$: timescale separation (**to recover**)
- $a$: excitability threshold (**to recover**)
- $b$: recovery sensitivity (known)
- $I_{\text{ext}}$: external stimulus current (known)

## Default Configuration

The generated template uses the following values.

**Parameters to recover:**

| Symbol | Code constant | True value |
|--------|---------------|------------|
| $\varepsilon$ | `TRUE_EPSILON` | $0.08$ |
| $a$ | `TRUE_A` | $0.7$ |

**Known constants:**

| Symbol | Code constant | Value |
|--------|---------------|-------|
| $b$ | `B` | $0.8$ |
| $I_{\text{ext}}$ | `I_EXT` | $0.5$ |

**Initial conditions:** $v(0) = -1.0, \quad w(0) = 1.0$

**Domain:** $t \in [0, 50]$

**Observability:** only $v$ (voltage) is observed; $w$ is a latent state reconstructed by the
network.

## Features Demonstrated

- Partial observability (only voltage is observed)
- Multi-parameter `Parameter` recovery ($\varepsilon$ and $a$)
- Neural excitation dynamics

## Results

![FitzHugh-Nagumo results](../examples/fitzhugh_nagumo/results/fitzhugh-nagumo.png)
