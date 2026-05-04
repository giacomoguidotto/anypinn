# Define a Custom Problem

This guide walks through defining your own ODE problem from scratch, starting
from the **Custom** or **Blank** template.

---

## Start from a template

Scaffold a project with the custom template:

```bash
anypinn create my-ode --template custom --data synthetic --lightning
```

This generates a skeleton with placeholder functions that you'll fill in.

---

## Write the ODE callable

The ODE callable is a function with this signature:

```python
def my_ode(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    ...
```

| Parameter | Shape | Meaning |
| --------- | ----- | ------- |
| `x` | `(m, 1)` | Independent variable (e.g. time) |
| `y` | `(n_fields, m, 1)` | Current state, one entry per field |
| `args` | `ArgsRegistry` | Named arguments and parameters |
| **returns** | `(n_fields, m, 1)` | Derivatives, one per field |

Every entry in `args` is callable: `args["beta"](x)` works for both fixed
`Argument`s and learnable `Parameter`s.

### Example: exponential decay

$$
\frac{dy}{dt} = -\lambda y
$$

```python
def exponential_decay(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    (Y,) = y
    lam = args["lambda"]
    dY = -lam(x) * Y
    return torch.stack([dY])
```

---

## Define fields and parameters

In `create_problem`, wire up `Field`s (neural network approximations of the
solution) and `Parameter`s (quantities to recover):

```python
from anypinn.core import Field, Parameter, Argument, FieldsRegistry, ParamsRegistry
from anypinn.problems import ODEInverseProblem, ODEProperties

def create_problem(hp):
    fields = FieldsRegistry({
        "Y": Field(config=hp.fields_config),
    })
    params = ParamsRegistry({
        "lambda": Parameter(config=hp.params_config),
    })

    props = ODEProperties(
        ode=exponential_decay,
        y0=torch.tensor([1.0]),       # Y(0) = 1
        args={},                       # No fixed arguments
    )

    return ODEInverseProblem(
        props=props,
        hp=hp,
        fields=fields,
        params=params,
    )
```

!!! tip "Fixed vs learnable"

    If `lambda` is known, use `Argument(0.5)` instead of `Parameter(...)` and
    move it into `props.args`. The ODE callable doesn't change, since both are
    accessed the same way via `args["lambda"](x)`.

---

## Set up validation

To track how well the PINN is recovering parameters during training, provide
ground-truth values:

```python
from anypinn.core import ValidationRegistry

validation: ValidationRegistry = {
    "lambda": 0.5,  # True value to compare against
}
```

The library logs MSE between recovered and ground-truth parameters at every
epoch.

For time-varying parameters, provide a callable:

```python
validation: ValidationRegistry = {
    "lambda": lambda t: 0.5 * torch.ones_like(t),
}
```

---

## Configure hyperparameters

In `config.py`, set up the network architecture and training parameters:

```python
hp = ODEHyperparameters(
    lr=1e-3,
    max_epochs=2000,
    training_data=GenerationConfig(
        batch_size=100,
        collocations=500,
        t_span=(0.0, 5.0),
        n_points=200,
    ),
    fields_config=MLPConfig(
        in_dim=1, out_dim=1,
        hidden_layers=[32, 64, 32],
        activation="tanh",
    ),
    params_config=ScalarConfig(init_value=0.1),
    pde_weight=1,
    ic_weight=10,
    data_weight=5,
)
```

!!! note "Scalar vs MLP parameters"

    Use `ScalarConfig` when the parameter is a constant (e.g. a rate
    coefficient). Use `MLPConfig` when the parameter is a function of the
    independent variable (e.g. a time-varying transmission rate).

---

## Train and inspect

```bash
uv sync && uv run train.py
```

Check the validation metrics in TensorBoard to see how the recovered parameter
converges to the true value.

---

## Higher-order ODEs

For second-order ODEs (or higher), set `order` in `ODEProperties`:

```python
props = ODEProperties(
    ode=my_second_order_ode,
    y0=torch.tensor([1.0]),
    dy0=[torch.tensor([0.0])],   # y'(0) = 0
    order=2,
)
```

The ODE callable receives an extra `derivs` argument:

```python
def my_second_order_ode(x, y, args, derivs=[]):
    (Y,) = y
    (dY,) = derivs[0]  # First derivative
    # Return d²Y/dt²
    return torch.stack([-args["omega"](x)**2 * Y])
```
