# Tune Hyperparameters

All training hyperparameters live in a single frozen dataclass in `config.py`.
This guide covers the most impactful knobs and when to turn them.

---

## The hyperparameter hierarchy

```python
hp = ODEHyperparameters(
    lr=5e-4,                          # Global learning rate
    max_epochs=2000,                  # Training duration
    gradient_clip_val=0.1,            # Gradient clipping threshold
    training_data=GenerationConfig(   # Data configuration
        batch_size=100,
        collocations=6000,
    ),
    fields_config=MLPConfig(...),     # Network for solution fields
    params_config=...,                # Network/scalar for parameters
    scheduler=ReduceLROnPlateauConfig(...),  # LR scheduling
    pde_weight=1,                     # Loss term weights
    ic_weight=1,
    data_weight=1,
)
```

---

## Learning rate

The single most impactful hyperparameter. Start with `1e-3` for simple problems
and decrease to `5e-4` or `1e-4` for harder ones.

!!! tip "Use a scheduler"

    `ReduceLROnPlateauConfig` automatically decreases the learning rate when
    the loss plateaus. This is almost always better than a fixed rate:

    ```python
    scheduler=ReduceLROnPlateauConfig(
        mode="min",
        factor=0.5,       # Halve the LR
        patience=55,      # Wait 55 epochs before reducing
        threshold=5e-3,   # Minimum improvement to count
        min_lr=1e-6,      # Don't go below this
    )
    ```

---

## Network architecture

### Fields (solution approximation)

```python
fields_config=MLPConfig(
    in_dim=1,                           # 1 for ODEs, 2+ for PDEs
    out_dim=1,                          # 1 per field
    hidden_layers=[64, 128, 128, 64],   # Width and depth
    activation="tanh",                  # Smooth activation for derivatives
    output_activation=None,             # None for unconstrained output
)
```

**Rules of thumb:**

- Start with `[64, 128, 128, 64]` — this works for most problems
- Add depth (more layers) for complex dynamics before adding width
- Use `tanh` as the default activation — it's smooth and works well with
  automatic differentiation
- Use `softplus` as output activation when the field must be positive

### Parameters

For **scalar parameters** (constants to recover):

```python
params_config=ScalarConfig(init_value=0.1)
```

For **function-valued parameters** (varying over the domain):

```python
params_config=MLPConfig(
    in_dim=1, out_dim=1,
    hidden_layers=[64, 128, 128, 64],
    activation="tanh",
    output_activation="softplus",     # Often needed to keep values positive
)
```

---

## Loss weights

The three weights `pde_weight`, `ic_weight`, and `data_weight` control the
balance between physics enforcement, initial conditions, and data fitting.

See [Loss Weighting Strategies](loss-weighting.md) for detailed guidance.

**Quick start:** Leave all weights at `1` initially, then increase `data_weight`
if the model doesn't fit observations, or increase `pde_weight` if the
recovered parameters are physically unreasonable.

---

## Collocation density

The number of collocation points controls how densely the physics is enforced:

```python
training_data=GenerationConfig(
    collocations=6000,     # More = better physics, slower training
    batch_size=100,        # Points per training step
)
```

- **Too few**: The network may satisfy the equation at sampled points but
  violate it elsewhere
- **Too many**: Training becomes slow with diminishing returns

Start with `1000–5000` for ODEs and `5000–20000` for PDEs.

---

## Gradient clipping

```python
gradient_clip_val=0.1
```

Prevents exploding gradients during early training. Lower values (0.01–0.1)
add stability at the cost of slower convergence. Set to `None` to disable.

---

## Diagnostic checklist

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| Loss doesn't decrease | Learning rate too low or too high | Try `1e-3`, `5e-4`, `1e-4` |
| Loss oscillates wildly | Learning rate too high | Reduce LR, increase `gradient_clip_val` |
| Good data fit, bad parameter recovery | `pde_weight` too low | Increase `pde_weight` |
| Smooth but wrong solution | Not enough collocation points | Increase `collocations` |
| Training is very slow | Network too wide or too many collocations | Reduce `hidden_layers` or `collocations` |
