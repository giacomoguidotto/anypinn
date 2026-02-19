"""Configuration dataclasses for PINN models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from torch import Tensor

from anypinn.core.types import Activations

if TYPE_CHECKING:
    from anypinn.core.nn import ArgsRegistry


@dataclass(kw_only=True)
class MLPConfig:
    """
    Configuration for a Multi-Layer Perceptron (MLP).

    Attributes:
        in_dim: Dimension of input layer.
        out_dim: Dimension of output layer.
        hidden_layers: List of dimensions for hidden layers.
        activation: Activation function to use between layers.
        output_activation: Optional activation function for the output layer.
        encode: Optional function to encode inputs before passing to MLP.
    """

    in_dim: int
    out_dim: int
    hidden_layers: list[int]
    activation: Activations
    output_activation: Activations | None = None
    encode: Callable[[Tensor], Tensor] | None = None


@dataclass(kw_only=True)
class ScalarConfig:
    """
    Configuration for a scalar parameter.

    Attributes:
        init_value: Initial value for the parameter.
    """

    init_value: float


@dataclass(kw_only=True)
class AdamConfig:
    """
    Configuration for the Adam optimizer.
    """

    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0


@dataclass(kw_only=True)
class LBFGSConfig:
    """
    Configuration for the L-BFGS optimizer.
    """

    lr: float = 1.0
    max_iter: int = 20
    max_eval: int | None = None
    history_size: int = 100
    line_search_fn: str | None = "strong_wolfe"


@dataclass(kw_only=True)
class ReduceLROnPlateauConfig:
    """
    Configuration for Learning Rate Scheduler (ReduceLROnPlateau).
    """

    mode: Literal["min", "max"]
    factor: float
    patience: int
    threshold: float
    min_lr: float


@dataclass(kw_only=True)
class CosineAnnealingConfig:
    """
    Configuration for Cosine Annealing LR Scheduler.
    """

    T_max: int
    eta_min: float = 0.0


@dataclass(kw_only=True)
class EarlyStoppingConfig:
    """
    Configuration for Early Stopping callback.
    """

    patience: int
    mode: Literal["min", "max"]


@dataclass(kw_only=True)
class SMMAStoppingConfig:
    """
    Configuration for Simple Moving Average Stopping callback.
    """

    window: int
    threshold: float
    lookback: int


@dataclass(kw_only=True)
class TrainingDataConfig:
    """
    Configuration for data loading and batching.
    """

    batch_size: int
    data_ratio: int | float
    collocations: int


@dataclass(kw_only=True)
class IngestionConfig(TrainingDataConfig):
    """
    Configuration for data ingestion from files.
    If x_column is None, the data is assumed to be evenly spaced.
    """

    df_path: Path
    x_transform: Callable[[Any], Any] | None = None
    x_column: str | None = None
    y_columns: list[str]


@dataclass(kw_only=True)
class GenerationConfig(TrainingDataConfig):
    """
    Configuration for data generation.
    """

    x: Tensor
    noise_level: float
    args_to_train: ArgsRegistry


@dataclass(kw_only=True)
class PINNHyperparameters:
    """
    Aggregated hyperparameters for the PINN model.
    """

    lr: float
    training_data: IngestionConfig | GenerationConfig
    fields_config: MLPConfig
    params_config: MLPConfig | ScalarConfig
    optimizer: AdamConfig | LBFGSConfig | None = None
    scheduler: ReduceLROnPlateauConfig | CosineAnnealingConfig | None = None
    early_stopping: EarlyStoppingConfig | None = None
    smma_stopping: SMMAStoppingConfig | None = None
