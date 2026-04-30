# Next Steps

You've trained your first PINN and understand the output. Here's where to go
depending on what you want to do.

---

## Modify the physics

Edit the ODE callable in `ode.py` to change the dynamics. Add or remove fields
and parameters in `create_problem`. The `ArgsRegistry` pattern means promoting
a fixed constant to a learnable parameter is a one-line change: replace
`Argument(value)` with `Parameter(config=hp.params_config)`.

[:octicons-arrow-right-24: Guide: Promote a constant to a parameter](../guides/promote-constant-to-parameter.md)

## Tune hyperparameters

Adjust learning rate, network architecture, loss weights, and collocation
density in `config.py`. The frozen dataclass ensures typos are caught
immediately.

[:octicons-arrow-right-24: Guide: Tune hyperparameters](../guides/tune-hyperparameters.md)

## Use your own data

Re-scaffold with `--data csv`, place your CSV in `data/`, and update the data
loading in `ode.py` to match your column names.

[:octicons-arrow-right-24: Guide: Use CSV data](../guides/csv-data.md)

## Try a different template

Each template demonstrates different AnyPINN features: Fourier encodings
(Lotka-Volterra), Huber loss (Lorenz), boundary conditions (Poisson 2D),
adaptive collocation (Burgers 1D). Scaffold several and compare how the ODE
definition and config differ.

[:octicons-arrow-right-24: Browse the catalog](../catalog/index.md)

## Define your own ODE

Start from the Custom or Blank template and write your own ODE callable,
fields, and parameters from scratch.

[:octicons-arrow-right-24: Guide: Define a custom ODE](../guides/custom-ode.md)

## Drop Lightning

If you need a non-standard training procedure, re-scaffold with
`--no-lightning`. The generated `train.py` gives you a raw PyTorch loop where
`problem` is a plain `nn.Module` and `problem.training_loss(batch, log)`
returns a scalar tensor.

[:octicons-arrow-right-24: Guide: Lightning vs Core](../guides/lightning-vs-core.md)

## Explore the API

The full API reference documents every public class and function.

[:octicons-arrow-right-24: API Reference](../reference/index.md)
