"""Configuration dataclasses for PINN models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from torch import Tensor

from anypinn.core.types import Activations, CollocationStrategies, Criteria

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

    Attributes:
        lr: Learning rate (must be positive).
        betas: Coefficients for computing running averages of gradient
            and its square. Both must be in (0, 1).
        weight_decay: L2 penalty coefficient (non-negative).
    """

    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}.")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}.")
        if not (0 < self.betas[0] < 1):
            raise ValueError(f"betas[0] must be in (0, 1), got {self.betas[0]}.")
        if not (0 < self.betas[1] < 1):
            raise ValueError(f"betas[1] must be in (0, 1), got {self.betas[1]}.")


@dataclass(kw_only=True)
class LBFGSConfig:
    """
    Configuration for the L-BFGS optimizer.

    Attributes:
        lr: Learning rate (must be positive).
        max_iter: Maximum number of iterations per optimization step.
        max_eval: Maximum number of function evaluations per step
            (defaults to ``max_iter * 1.25``).
        history_size: Number of past updates to store for the
            approximation of the inverse Hessian.
        line_search_fn: Line search function (``"strong_wolfe"`` or ``None``).
    """

    lr: float = 1.0
    max_iter: int = 20
    max_eval: int | None = None
    history_size: int = 100
    line_search_fn: str | None = "strong_wolfe"

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}.")
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {self.max_iter}.")
        if self.history_size <= 0:
            raise ValueError(f"history_size must be positive, got {self.history_size}.")


@dataclass(kw_only=True)
class ReduceLROnPlateauConfig:
    """
    Configuration for Learning Rate Scheduler (ReduceLROnPlateau).

    Attributes:
        mode: ``"min"`` to reduce LR when the metric stops decreasing,
            ``"max"`` when it stops increasing.
        factor: Factor by which the learning rate is reduced (must be
            in (0, 1)).
        patience: Number of epochs with no improvement before the LR
            is reduced.
        threshold: Minimum change to qualify as an improvement.
        min_lr: Lower bound on the learning rate.
    """

    mode: Literal["min", "max"]
    factor: float
    patience: int
    threshold: float
    min_lr: float

    def __post_init__(self) -> None:
        if not (0 < self.factor < 1):
            raise ValueError(f"factor must be in (0, 1), got {self.factor}.")
        if self.patience <= 0:
            raise ValueError(f"patience must be positive, got {self.patience}.")


@dataclass(kw_only=True)
class CosineAnnealingConfig:
    """
    Configuration for Cosine Annealing LR Scheduler.

    Attributes:
        T_max: Maximum number of iterations (typically set to
            ``max_epochs``).
        eta_min: Minimum learning rate at the end of the schedule.
    """

    T_max: int
    eta_min: float = 0.0

    def __post_init__(self) -> None:
        if self.T_max <= 0:
            raise ValueError(f"T_max must be positive, got {self.T_max}.")


@dataclass(kw_only=True)
class EarlyStoppingConfig:
    """
    Configuration for Early Stopping callback.

    Attributes:
        patience: Number of epochs with no improvement before stopping.
        mode: ``"min"`` to stop when the metric stops decreasing,
            ``"max"`` when it stops increasing.
    """

    patience: int
    mode: Literal["min", "max"]

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError(f"patience must be positive, got {self.patience}.")


@dataclass(kw_only=True)
class SMMAStoppingConfig:
    """
    Configuration for Smoothed Moving Average (SMMA) Stopping callback.

    Training stops when the relative improvement of the SMMA over the
    ``lookback`` window falls below ``threshold``.

    Attributes:
        window: Number of epochs used to compute the smoothed moving
            average.
        threshold: Minimum relative improvement required to continue
            training.
        lookback: Number of SMMA values to compare for improvement
            detection.
    """

    window: int
    threshold: float
    lookback: int

    def __post_init__(self) -> None:
        if self.window <= 0:
            raise ValueError(f"window must be positive, got {self.window}.")
        if self.lookback <= 0:
            raise ValueError(f"lookback must be positive, got {self.lookback}.")
        if self.threshold <= 0:
            raise ValueError(f"threshold must be positive, got {self.threshold}.")


@dataclass(kw_only=True)
class TrainingDataConfig:
    """
    Configuration for data loading and batching.

    Attributes:
        batch_size: Number of points per training batch.
        data_ratio: Ratio of data to collocation points per batch.
        collocations: Total number of collocation points to generate.
        collocation_sampler: Sampling strategy for collocation points.
        collocation_seed: Optional seed for reproducible collocation sampling.
    """

    batch_size: int
    data_ratio: int | float
    collocations: int
    collocation_sampler: CollocationStrategies = "random"
    collocation_seed: int | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}.")
        if self.collocations < 0:
            raise ValueError(f"collocations must be non-negative, got {self.collocations}.")
        if isinstance(self.data_ratio, float):
            if not (0.0 <= self.data_ratio <= 1.0):
                raise ValueError(f"Float data_ratio must be in [0.0, 1.0], got {self.data_ratio}.")
        else:
            if not (0 <= self.data_ratio <= self.batch_size):
                raise ValueError(
                    f"Integer data_ratio must be in [0, {self.batch_size}], got {self.data_ratio}."
                )


@dataclass(kw_only=True)
class IngestionConfig(TrainingDataConfig):
    """
    Configuration for loading training data from a CSV file.

    Attributes:
        df_path: Path to the CSV file.
        x_transform: Optional transform applied to the x column values
            after loading (e.g. unit conversion).
        x_column: Name of the column to use as x coordinates. If
            ``None``, rows are assumed to be evenly spaced and an
            integer index is used.
        y_columns: List of column names to use as y observations.
    """

    df_path: Path
    x_transform: Callable[[Any], Any] | None = None
    x_column: str | None = None
    y_columns: list[str]


@dataclass(kw_only=True)
class GenerationConfig(TrainingDataConfig):
    """
    Configuration for generating synthetic training data.

    Used in forward problems where the ground-truth ODE/PDE solution is
    computed from known parameters and optionally corrupted with noise.

    Attributes:
        x: Coordinate tensor to evaluate the ODE/PDE at.
        noise_level: Standard deviation of Gaussian noise added to the
            generated observations (0.0 for clean data).
        args_to_train: Arguments used by the data-generation ODE/PDE
            callable to produce the synthetic solution.
    """

    x: Tensor
    noise_level: float
    args_to_train: ArgsRegistry


@dataclass(kw_only=True)
class PINNHyperparameters:
    """
    Aggregated hyperparameters for the PINN model.

    Attributes:
        lr: Base learning rate (used as fallback when no ``optimizer``
            config is provided).
        training_data: Data source configuration — either
            ``IngestionConfig`` (CSV) or ``GenerationConfig`` (synthetic).
        fields_config: MLP architecture for the neural field(s).
        params_config: Configuration for learnable parameters (scalar
            or MLP-backed).
        max_epochs: Maximum number of training epochs.
        gradient_clip_val: Optional gradient clipping value.
        criterion: Loss function name (``"mse"``, ``"huber"``, or
            ``"l1"``).
        optimizer: Optimizer configuration. If ``None``, Adam with
            ``lr`` is used.
        scheduler: Learning rate scheduler configuration.
        early_stopping: Optional early stopping configuration
            (patience-based).
        smma_stopping: Optional SMMA stopping configuration
            (improvement-based).
    """

    lr: float
    training_data: IngestionConfig | GenerationConfig
    fields_config: MLPConfig
    params_config: MLPConfig | ScalarConfig
    max_epochs: int | None = None
    gradient_clip_val: float | None = None
    criterion: Criteria = "mse"
    optimizer: AdamConfig | LBFGSConfig | None = None
    scheduler: ReduceLROnPlateauConfig | CosineAnnealingConfig | None = None
    early_stopping: EarlyStoppingConfig | None = None
    smma_stopping: SMMAStoppingConfig | None = None

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}.")
