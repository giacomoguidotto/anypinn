# Core API

`anypinn.core` provides the foundational building blocks for defining and
solving physics-informed neural network problems.

---

## Neural Network Primitives

### Domain

::: anypinn.core.Domain
    options:
      show_source: false
      show_signature_annotations: true

### Field

::: anypinn.core.Field
    options:
      show_source: false
      show_signature_annotations: true

### Argument

::: anypinn.core.Argument
    options:
      show_source: false
      show_signature_annotations: true

### Parameter

::: anypinn.core.Parameter
    options:
      show_source: false
      show_signature_annotations: true

---

## Problem Abstractions

### Constraint

::: anypinn.core.Constraint
    options:
      show_source: false
      show_signature_annotations: true

### Problem

::: anypinn.core.Problem
    options:
      show_source: false
      show_signature_annotations: true

---

## Configuration

### MLPConfig

::: anypinn.core.MLPConfig
    options:
      show_source: false
      show_signature_annotations: true

### ScalarConfig

::: anypinn.core.ScalarConfig
    options:
      show_source: false
      show_signature_annotations: true

### PINNHyperparameters

::: anypinn.core.PINNHyperparameters
    options:
      show_source: false
      show_signature_annotations: true

### TrainingDataConfig

::: anypinn.core.TrainingDataConfig
    options:
      show_source: false
      show_signature_annotations: true

### GenerationConfig

::: anypinn.core.GenerationConfig
    options:
      show_source: false
      show_signature_annotations: true

### IngestionConfig

::: anypinn.core.IngestionConfig
    options:
      show_source: false
      show_signature_annotations: true

### Optimizer Configs

::: anypinn.core.AdamConfig
    options:
      show_source: false
      show_signature_annotations: true

::: anypinn.core.LBFGSConfig
    options:
      show_source: false
      show_signature_annotations: true

### Scheduler Configs

::: anypinn.core.ReduceLROnPlateauConfig
    options:
      show_source: false
      show_signature_annotations: true

::: anypinn.core.CosineAnnealingConfig
    options:
      show_source: false
      show_signature_annotations: true

### Stopping Configs

::: anypinn.core.EarlyStoppingConfig
    options:
      show_source: false
      show_signature_annotations: true

::: anypinn.core.SMMAStoppingConfig
    options:
      show_source: false
      show_signature_annotations: true

---

## Collocation Samplers

::: anypinn.core.CollocationSampler
    options:
      show_source: false
      show_signature_annotations: true

::: anypinn.core.UniformSampler
    options:
      show_source: false
      show_signature_annotations: true

::: anypinn.core.RandomSampler
    options:
      show_source: false
      show_signature_annotations: true

::: anypinn.core.LatinHypercubeSampler
    options:
      show_source: false
      show_signature_annotations: true

::: anypinn.core.LogUniform1DSampler
    options:
      show_source: false
      show_signature_annotations: true

::: anypinn.core.AdaptiveSampler
    options:
      show_source: false
      show_signature_annotations: true

::: anypinn.core.ResidualScorer
    options:
      show_source: false
      show_signature_annotations: true

::: anypinn.core.build_sampler
    options:
      show_source: false
      show_signature_annotations: true

---

## Data Handling

### PINNDataset

::: anypinn.core.PINNDataset
    options:
      show_source: false
      show_signature_annotations: true

### PINNDataModule

::: anypinn.core.PINNDataModule
    options:
      show_source: false
      show_signature_annotations: true

### DataCallback

::: anypinn.core.DataCallback
    options:
      show_source: false
      show_signature_annotations: true

### InferredContext

::: anypinn.core.InferredContext
    options:
      show_source: false
      show_signature_annotations: true

---

## Validation

::: anypinn.core.ColumnRef
    options:
      show_source: false
      show_signature_annotations: true

::: anypinn.core.resolve_validation
    options:
      show_source: false
      show_signature_annotations: true

---

## Encodings

::: anypinn.core.FourierEncoding
    options:
      show_source: false
      show_signature_annotations: true

::: anypinn.core.RandomFourierFeatures
    options:
      show_source: false
      show_signature_annotations: true

---

## Utility Functions

::: anypinn.core.build_criterion
    options:
      show_source: false
      show_signature_annotations: true

::: anypinn.core.get_activation
    options:
      show_source: false
      show_signature_annotations: true

---

## Type Aliases

| Alias | Definition | Purpose |
| ----- | ---------- | ------- |
| `ArgsRegistry` | `dict[str, Argument]` | Named arguments passed to ODE/PDE callables |
| `ParamsRegistry` | `dict[str, Parameter]` | Named learnable parameters |
| `FieldsRegistry` | `dict[str, Field]` | Named neural fields |
| `TrainingBatch` | `tuple[DataBatch, Tensor]` | `(data, collocation_points)` |
| `DataBatch` | `tuple[Tensor, Tensor]` | `(x_data, y_data)` |
| `Predictions` | `tuple[DataBatch, dict, dict \| None]` | `(batch, preds, trues)` |
| `LogFn` | `Protocol` | Logging callback `(key, value, progress_bar?)` |
| `ValidationRegistry` | `dict[str, ValidationSource]` | Ground-truth values for parameter comparison |
| `Activations` | `Literal[...]` | Supported activation function names |
| `Criteria` | `Literal[...]` | Supported loss function names |
| `CollocationStrategies` | `Literal[...]` | Supported collocation sampling strategies |
