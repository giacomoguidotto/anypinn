# PINN Primer

A short conceptual introduction to Physics-Informed Neural Networks for readers
who haven't encountered them before.

---

## The idea

A Physics-Informed Neural Network (PINN) is a neural network trained to
satisfy a differential equation. Instead of learning purely from data, the
network's loss function includes a **physics term** that penalizes violations
of the governing equations.

This means a PINN can:

- **Solve differential equations** without a mesh or numerical integrator
- **Recover unknown parameters** from partial observations (inverse problems)
- **Interpolate between sparse measurements** while respecting physical laws

---

## How it works

Consider a simple ODE:

$$
\frac{dy}{dt} = f(t, y, \theta)
$$

where \(\theta\) is an unknown parameter we want to recover from data.

A PINN replaces the solution \(y(t)\) with a neural network \(y_\text{NN}(t)\)
and trains it by minimizing a composite loss:

$$
\mathcal{L} = \underbrace{w_r \left\| \frac{dy_\text{NN}}{dt} - f(t, y_\text{NN}, \theta) \right\|^2}_{\text{Residual loss}}
+ \underbrace{w_{\text{ic}} \left\| y_\text{NN}(t_0) - y_0 \right\|^2}_{\text{Initial condition loss}}
+ \underbrace{w_d \left\| y_\text{NN}(t_i) - y_i^{\text{obs}} \right\|^2}_{\text{Data loss}}
$$

| Term | Purpose | Where evaluated |
| ---- | ------- | --------------- |
| **Residual loss** | Enforce the differential equation | Collocation points (sampled across the domain) |
| **IC loss** | Enforce initial conditions | The initial time point |
| **Data loss** | Fit observed measurements | Data points |

The key insight: the residual loss uses **automatic differentiation** to compute
\(dy_\text{NN}/dt\) exactly — no finite differences, no discretization error.
The physics is enforced continuously across the domain, not just at grid points.

---

## Forward vs Inverse

**Forward problem**: The equation and all parameters are known. The PINN
approximates the solution \(y(t)\).

**Inverse problem**: Some parameters \(\theta\) are unknown. The PINN
simultaneously approximates the solution *and* recovers \(\theta\) from
partial observations. This is AnyPINN's primary use case.

[:octicons-arrow-right-24: More on forward vs inverse](../guides/inverse-vs-forward.md)

---

## Collocation points

The residual loss is evaluated at **collocation points** — coordinates sampled
across the domain where the differential equation must hold. More collocation
points means better enforcement of the physics, but also more computation per
training step.

AnyPINN supports several sampling strategies:

- **Uniform** — regular grid
- **Random** — uniform random
- **Latin Hypercube** — space-filling quasi-random
- **Adaptive** — residual-weighted, concentrating points where the equation is
  hardest to satisfy

---

## Loss weights

The three loss terms compete during training. The weights
\(w_r, w_{\text{ic}}, w_d\) control the balance:

- High `pde_weight` → stronger physics enforcement, may underfit data
- High `data_weight` → closer fit to observations, may violate the equation
- High `ic_weight` → tighter initial condition match

Finding the right balance is problem-specific. See the
[loss weighting guide](../guides/loss-weighting.md) for practical advice.

---

## Where AnyPINN fits

AnyPINN handles the wiring for you. You write the ODE/PDE callable —
the mathematical definition of your problem — and AnyPINN constructs the
composite loss, manages fields and parameters, handles collocation sampling,
and provides validation against ground truth. Training is delegated to
PyTorch or PyTorch Lightning.

[:octicons-arrow-right-24: Continue to Installation](installation.md)
