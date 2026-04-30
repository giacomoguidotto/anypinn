# API Reference

Complete reference for all public classes and functions in AnyPINN, organized
by module.

---

<div class="grid cards" markdown>

- :material-cube-outline: **[Core](core.md)**

    ---

    Building blocks: `Domain`, `Field`, `Parameter`, `Argument`, `Constraint`,
    `Problem`, configuration dataclasses, collocation samplers, and data
    handling.

- :material-function-variant: **[Problems](problems.md)**

    ---

    ODE and PDE constraint types: `ODEInverseProblem`, `ResidualsConstraint`,
    `DirichletBCConstraint`, `PDEResidualConstraint`, and supporting classes.

- :material-lightning-bolt: **[Lightning](lightning.md)**

    ---

    PyTorch Lightning integration: `PINNModule`, callbacks for early stopping,
    progress bars, prediction writing, and adaptive collocation.

- :material-console: **[CLI](cli.md)**

    ---

    Command-line interface reference: `anypinn create`, template listing,
    data source selection, and project scaffolding options.

</div>
