# Loss Weighting Strategies

The PINN loss is a weighted sum of competing objectives. Getting the balance
right is often the difference between convergence and failure.

---

## The composite loss

$$
\mathcal{L} = w_r \mathcal{L}_{\text{residual}}
+ w_{\text{ic}} \mathcal{L}_{\text{IC}}
+ w_d \mathcal{L}_{\text{data}}
$$

In `config.py`:

```python
hp = ODEHyperparameters(
    ...
    pde_weight=1,      # w_r:  physics enforcement
    ic_weight=1,       # w_ic: initial condition
    data_weight=1,     # w_d:  data fitting
)
```

---

## What each weight controls

### `pde_weight` (residual)

How strongly the differential equation is enforced at collocation points.

- **Too low**: The network ignores physics, overfitting to data. Recovered
  parameters may be physically meaningless.
- **Too high**: The network satisfies the equation but doesn't fit the data.
  Parameter recovery stalls.

### `ic_weight` (initial conditions)

How tightly the initial state is enforced.

- **Too low**: The solution drifts from the correct starting point.
- **Too high**: Training focuses on the initial condition at the expense of
  the rest of the domain.

### `data_weight` (observations)

How closely the predicted fields match observed data.

- **Too low**: The network doesn't learn from observations. For inverse
  problems, parameter recovery fails.
- **Too high**: The network overfits to noise in the data, and the physics
  constraint becomes ineffective.

---

## Practical strategies

### Start with equal weights

```python
pde_weight=1, ic_weight=1, data_weight=1
```

This is a reasonable starting point for most problems. Check TensorBoard to see
which loss term dominates.

### Scale by magnitude

If the loss terms have very different magnitudes (e.g. residual loss is 1e-3
but data loss is 1e2), the larger term drowns out the smaller one. Scale the
weights to compensate:

```python
# Residual loss ≈ 0.001, data loss ≈ 100
pde_weight=1000, ic_weight=1, data_weight=1
```

### Increase `pde_weight` for parameter recovery

If the data fit is good but the recovered parameters are wrong, the network is
finding a shortcut that satisfies the data without following the physics. Fix
this by increasing `pde_weight`:

```python
pde_weight=10, ic_weight=1, data_weight=1
```

### Increase `ic_weight` for early-time accuracy

If the solution is accurate at later times but wrong near the initial
condition, increase `ic_weight`:

```python
pde_weight=1, ic_weight=10, data_weight=1
```

---

## Diagnostic workflow

1. **Train with equal weights** and check TensorBoard
2. **Compare loss magnitudes**: if one term is orders of magnitude larger, it's
   dominating the optimization
3. **Check parameter validation**: if `val/*_mse` isn't decreasing, the physics
   constraint isn't strong enough
4. **Check data fit**: if the predicted fields don't match observations, the
   data weight is too low

| Symptom | Diagnosis | Action |
| ------- | --------- | ------ |
| Good data fit, bad parameters | Physics too weak | Increase `pde_weight` |
| Bad data fit, smooth solution | Data too weak | Increase `data_weight` |
| Wrong near t=0, fine later | IC too weak | Increase `ic_weight` |
| Nothing converges | Imbalanced magnitudes | Normalize loss terms, check learning rate |

---

## Advanced: per-constraint weights

For PDE problems with multiple boundary conditions, each constraint has its own
`weight` parameter:

```python
dirichlet_left = DirichletBCConstraint(bc=bc_left, field=u, weight=10.0)
dirichlet_right = DirichletBCConstraint(bc=bc_right, field=u, weight=5.0)
residual = PDEResidualConstraint(residual_fn=pde, fields=fields, weight=1.0)
```

This gives fine-grained control when some boundaries are harder to satisfy
than others.
