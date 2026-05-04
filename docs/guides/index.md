# Guides

Task-oriented guides for common workflows. Each guide solves a specific
problem and assumes you've completed the
[Getting Started](../getting-started/index.md) tutorial.

---

## Defining Problems

<div class="grid cards" markdown>

- :material-function-variant: **[Define a Custom Problem](custom.md)**

    Write your own ODE callable, fields, and parameters from scratch.

- :material-swap-horizontal: **[Promote a Constant to a Parameter](promote-constant-to-parameter.md)**

    Turn a fixed value into a learnable quantity with a one-line change.

- :material-arrow-split-vertical: **[Forward vs Inverse Problems](inverse-vs-forward.md)**

    Understand when to use each mode and how the loss structure differs.

- :material-grid: **[PDE Forward Problems](pde-forward-problems.md)**

    Set up boundary conditions, multi-dimensional domains, and PDE residuals.

</div>

## Training and Data

<div class="grid cards" markdown>

- :material-file-delimited: **[Use CSV Data](csv-data.md)**

    Load real experimental observations instead of synthetic data.

- :material-tune-vertical: **[Tune Hyperparameters](tune-hyperparameters.md)**

    Adjust learning rate, architecture, loss weights, and collocation density.

- :material-scale-balance: **[Loss Weighting Strategies](loss-weighting.md)**

    Balance physics enforcement against data fitting.

- :material-lightning-bolt: **[Lightning vs Core Training](lightning-vs-core.md)**

    Choose between PyTorch Lightning and a raw training loop.

</div>

## Understanding the Design

<div class="grid cards" markdown>

- :material-layers-triple: **[Architecture](architecture.md)**

    How AnyPINN's layered design compares to other PINN libraries.

</div>
