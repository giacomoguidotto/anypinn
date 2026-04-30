# Promote a Constant to a Parameter

One of AnyPINN's key design choices is that fixed constants and learnable
parameters share the same interface. This guide shows how to turn a known
value into something the network recovers — a one-line change.

---

## The pattern

In every ODE callable, arguments are accessed the same way regardless of
whether they're fixed or learnable:

```python
def my_ode(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    beta = args["beta"]    # Could be Argument or Parameter
    value = beta(x)        # Both are callable: Argument(v)(x) returns v
    ...
```

This uniformity means the ODE function never needs to change when you switch
a constant to a learnable parameter.

---

## Before: fixed constant

```python
from anypinn.core import Argument

props = ODEProperties(
    ode=my_ode,
    y0=torch.tensor([...]),
    args={
        "beta": Argument(0.3),      # Fixed at 0.3
        "gamma": Argument(0.1),     # Fixed at 0.1
    },
)

# create_problem — no params for beta
params = ParamsRegistry({})
```

## After: learnable parameter

```python
from anypinn.core import Argument, Parameter

props = ODEProperties(
    ode=my_ode,
    y0=torch.tensor([...]),
    args={
        # "beta" removed from fixed args
        "gamma": Argument(0.1),     # Still fixed
    },
)

# create_problem — beta is now learnable
params = ParamsRegistry({
    "beta": Parameter(config=hp.params_config),  # (1)!
})
```

1. `hp.params_config` can be `ScalarConfig(init_value=0.1)` for a constant
   parameter or `MLPConfig(...)` for a function-valued parameter.

That's it. The ODE callable, the fields, and the training script are unchanged.

---

## Choosing scalar vs function-valued

| Config | When to use | What it learns |
| ------ | ----------- | -------------- |
| `ScalarConfig(init_value=0.1)` | The parameter is a single number (e.g. a rate constant) | One scalar value |
| `MLPConfig(in_dim=1, out_dim=1, ...)` | The parameter varies over the domain (e.g. a time-varying coefficient) | A function `θ(x)` |

Both produce a callable that takes `x` and returns a tensor, so the ODE
function works identically either way.

---

## Adding validation

When you promote a constant, add it to the `ValidationRegistry` so training
logs show how well the recovery is going:

```python
validation: ValidationRegistry = {
    "beta": 0.3,  # The true value you just removed from args
}
```

For function-valued parameters, provide the ground-truth function:

```python
validation: ValidationRegistry = {
    "beta": lambda t: 0.3 * torch.ones_like(t),
}
```
