"""Core PINN building blocks."""

from anypinn.core.config import (
    AdamConfig,
    CosineAnnealingConfig,
    EarlyStoppingConfig,
    GenerationConfig,
    IngestionConfig,
    LBFGSConfig,
    MLPConfig,
    PINNHyperparameters,
    ReduceLROnPlateauConfig,
    ScalarConfig,
    SMMAStoppingConfig,
    TrainingDataConfig,
)
from anypinn.core.context import InferredContext
from anypinn.core.dataset import DataCallback, PINNDataModule, PINNDataset
from anypinn.core.nn import (
    ArgsRegistry,
    Argument,
    Domain1D,
    Field,
    FieldsRegistry,
    Parameter,
    ParamsRegistry,
    get_activation,
)
from anypinn.core.problem import Constraint, Problem
from anypinn.core.types import LOSS_KEY, Activations, DataBatch, LogFn, Predictions, TrainingBatch
from anypinn.core.validation import (
    ColumnRef,
    ResolvedValidation,
    ValidationRegistry,
    ValidationSource,
    resolve_validation,
)

__all__ = [
    "LOSS_KEY",
    "Activations",
    "AdamConfig",
    "ArgsRegistry",
    "Argument",
    "ColumnRef",
    "Constraint",
    "CosineAnnealingConfig",
    "DataBatch",
    "DataCallback",
    "Domain1D",
    "EarlyStoppingConfig",
    "Field",
    "FieldsRegistry",
    "GenerationConfig",
    "InferredContext",
    "IngestionConfig",
    "LBFGSConfig",
    "LogFn",
    "MLPConfig",
    "PINNDataModule",
    "PINNDataset",
    "PINNHyperparameters",
    "Parameter",
    "ParamsRegistry",
    "Predictions",
    "Problem",
    "ReduceLROnPlateauConfig",
    "ResolvedValidation",
    "SMMAStoppingConfig",
    "ScalarConfig",
    "TrainingBatch",
    "TrainingDataConfig",
    "ValidationRegistry",
    "ValidationSource",
    "get_activation",
    "resolve_validation",
]
