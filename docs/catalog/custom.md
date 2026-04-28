# Custom ODE

```bash
anypinn create my-project --template custom
```

Minimal skeleton for a user-defined ODE. All factories (`create_problem`, `create_data_module`) are
stubs ready to be filled in with your own physics.

## When to Use

Use this template when your problem is not covered by the built-in templates but you want the
standard project structure (`ode.py`, `config.py`, `train.py`) as a starting point.

## What to Modify

1. Define your ODE callable in `ode.py`
2. Set up `Field`s and `Parameter`s in `create_problem`
3. Configure hyperparameters in `config.py`
