# Problems API

`anypinn.problems` provides ready-made constraint types for ODE and PDE
problems, plus convenience classes that wire them together.

---

## ODE Constraints

### ODECallable

::: anypinn.problems.ODECallable
    options:
      show_source: false
      show_signature_annotations: true

### ODEProperties

::: anypinn.problems.ODEProperties
    options:
      show_source: false
      show_signature_annotations: true

### ResidualsConstraint

::: anypinn.problems.ResidualsConstraint
    options:
      show_source: false
      show_signature_annotations: true

### ICConstraint

::: anypinn.problems.ICConstraint
    options:
      show_source: false
      show_signature_annotations: true

### DataConstraint

::: anypinn.problems.DataConstraint
    options:
      show_source: false
      show_signature_annotations: true

### ODEInverseProblem

::: anypinn.problems.ODEInverseProblem
    options:
      show_source: false
      show_signature_annotations: true

### ODEHyperparameters

::: anypinn.problems.ODEHyperparameters
    options:
      show_source: false
      show_signature_annotations: true

---

## PDE Constraints

### BoundaryCondition

::: anypinn.problems.BoundaryCondition
    options:
      show_source: false
      show_signature_annotations: true

### DirichletBCConstraint

::: anypinn.problems.DirichletBCConstraint
    options:
      show_source: false
      show_signature_annotations: true

### NeumannBCConstraint

::: anypinn.problems.NeumannBCConstraint
    options:
      show_source: false
      show_signature_annotations: true

### PeriodicBCConstraint

::: anypinn.problems.PeriodicBCConstraint
    options:
      show_source: false
      show_signature_annotations: true

### PDEResidualConstraint

::: anypinn.problems.PDEResidualConstraint
    options:
      show_source: false
      show_signature_annotations: true

---

## Type Aliases

| Alias | Definition | Purpose |
| ----- | ---------- | ------- |
| `PredictDataFn` | `Callable[[Tensor, FieldsRegistry, ParamsRegistry], Tensor]` | Maps fields/params to observed quantities |
| `BCValueFn` | `Callable[[Tensor], Tensor]` | Boundary value function |
| `PDEResidualFn` | `Callable[[Tensor, FieldsRegistry, ParamsRegistry], Tensor]` | PDE residual function |
