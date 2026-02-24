"""Data handling for PINN training."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from os import cpu_count
from typing import cast, override

import lightning as pl
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

from anypinn.core.config import GenerationConfig, IngestionConfig, PINNHyperparameters
from anypinn.core.context import InferredContext
from anypinn.core.nn import Domain
from anypinn.core.samplers import CollocationSampler, ResidualScorer, build_sampler
from anypinn.core.types import CollocationStrategies, DataBatch, PredictionBatch, TrainingBatch
from anypinn.core.validation import ValidationRegistry, resolve_validation


class DataCallback:
    """Abstract base class for building new data callbacks."""

    def transform_data(self, data: DataBatch, coll: Tensor) -> tuple[DataBatch, Tensor]:
        """Transform the data and collocation points."""
        return data, coll

    def on_after_setup(self, dm: "PINNDataModule") -> None:
        """Called after setup is complete."""
        return None


class PINNDataset(Dataset[TrainingBatch]):
    """
    Dataset used for PINN training. Combines labeled data and collocation points
    per sample.  Given a data_ratio, the amount of data points `K` is determined
    either by applying `data_ratio * batch_size` if ratio is a float between 0
    and 1 or by an absolute count if ratio is an integer. The remaining `C`
    points are used for collocation.  The data points are sampled without
    replacement per epoch i.e. cycles through all data points and at the last
    batch, wraps around to the first indices to ensure batch size. The collocation
    points are sampled with replacement from the pool.
    The dataset produces a batch of shape ((x_data[K,d], y_data[K,...]), x_coll[C,d]).

    Args:
        x_data: Data point x coordinates (time values).
        y_data: Data point y values (observations).
        x_coll: Collocation point x coordinates.
        batch_size: Size of the batch.
        data_ratio: Ratio of data points to collocation points, either as a ratio [0,1] or absolute
            count [0,batch_size].
    """

    def __init__(
        self,
        x_data: Tensor,
        y_data: Tensor,
        x_coll: Tensor,
        batch_size: int,
        data_ratio: float | int,
    ):
        super().__init__()
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        if isinstance(data_ratio, float):
            if not (0.0 <= data_ratio <= 1.0):
                raise ValueError(f"Float data_ratio must be in [0.0, 1.0], got {data_ratio}.")
            self.K = round(data_ratio * batch_size)
        else:
            if not (0 <= data_ratio <= batch_size):
                raise ValueError(
                    f"Integer data_ratio must be in [0, {batch_size}], got {data_ratio}."
                )
            self.K = data_ratio

        self.x_data = x_data
        self.y_data = y_data
        self.x_coll = x_coll

        self.batch_size = batch_size
        self.C = batch_size - self.K

        self.total_data = x_data.shape[0]
        self.total_coll = x_coll.shape[0]

        self._coll_gen = torch.Generator()

    def __len__(self) -> int:
        """Number of steps per epoch to see all data points once. Ceiling division."""
        return (self.total_data + self.K - 1) // self.K

    @override
    def __getitem__(self, index: int) -> TrainingBatch:
        """Return one sample containing K data points and C collocation points."""
        data_idx = self._get_data_indices(index)
        coll_idx = self._get_coll_indices(index)

        x_data = self.x_data[data_idx]
        y_data = self.y_data[data_idx]
        x_coll = self.x_coll[coll_idx]

        return ((x_data, y_data), x_coll)

    def _get_data_indices(self, idx: int) -> Tensor:
        """Get data indices for this step without replacement.
        When getting the last batch, wrap around to the first indices to ensure batch size.
        """
        if self.total_data == 0:
            return torch.empty(0, 1)

        start = idx * self.K
        indices = [(start + i) % self.total_data for i in range(self.K)]
        return torch.tensor(indices)

    def _get_coll_indices(self, idx: int) -> Tensor:
        """Get collocation indices for this step with replacement."""
        if self.total_coll == 0:
            return torch.empty(0, 1)

        self._coll_gen.manual_seed(idx)
        return torch.randint(0, self.total_coll, (self.C,), generator=self._coll_gen)


class PINNDataModule(pl.LightningDataModule, ABC):
    """
    LightningDataModule for PINNs.
    Manages data and collocation datasets and creates the combined PINNDataset.

    Collocation points are generated via a ``CollocationSampler`` selected by the
    ``collocation_sampler`` field in ``TrainingDataConfig`` (string literal).
    Subclasses only need to implement ``gen_data()``; collocation generation is
    handled by the sampler resolved from the hyperparameters.

    Attributes:
        pinn_ds: Combined PINNDataset for training.
        callbacks: Sequence of DataCallback callbacks applied after data loading.
    """

    def __init__(
        self,
        hp: PINNHyperparameters,
        validation: ValidationRegistry | None = None,
        callbacks: Sequence[DataCallback] | None = None,
        residual_scorer: ResidualScorer | None = None,
    ) -> None:
        super().__init__()
        self.hp = hp
        self.callbacks: list[DataCallback] = list(callbacks) if callbacks else []
        self._residual_scorer = residual_scorer

        self._unresolved_validation = validation or {}
        self._context: InferredContext | None = None

    def _build_sampler(self, strategy: CollocationStrategies) -> CollocationSampler:
        """Resolve a collocation sampler from a strategy name."""
        return build_sampler(
            strategy=strategy,
            seed=self.hp.training_data.collocation_seed,
            scorer=self._residual_scorer,
        )

    def load_data(self, config: IngestionConfig) -> DataBatch:
        """Load raw data from IngestionConfig."""
        df = pd.read_csv(config.df_path)

        if config.x_column is not None:
            x_values = df[config.x_column].values

            if config.x_transform is not None:
                x_values = config.x_transform(x_values)

            x = torch.tensor(x_values, dtype=torch.float32)
        else:
            x = torch.arange(len(df), dtype=torch.float32)

        y = torch.tensor(df[config.y_columns].values, dtype=torch.float32)

        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] > 1):
            y = y.unsqueeze(-1)

        return x.unsqueeze(-1), y

    @abstractmethod
    def gen_data(self, config: GenerationConfig) -> DataBatch:
        """Generate synthetic data from GenerationConfig."""

    @override
    def setup(self, stage: str | None = None) -> None:
        """
        Load raw data from IngestionConfig, or generate synthetic data from GenerationConfig.
        Apply registered callbacks, create InferredContext and datasets.
        """
        config = self.hp.training_data

        self.validation = resolve_validation(
            self._unresolved_validation,
            config.df_path if isinstance(config, IngestionConfig) else None,
        )

        self.data = (
            self.load_data(config)
            if isinstance(config, IngestionConfig)
            else self.gen_data(config)
        )

        domain = Domain.from_x(self.data[0])
        self._domain = domain
        self._sampler = self._build_sampler(config.collocation_sampler)
        self.coll = self._sampler.sample(config.collocations, domain)

        for callback in self.callbacks:
            self.data, self.coll = callback.transform_data(self.data, self.coll)

        x_data, y_data = self.data

        if x_data.shape[0] != y_data.shape[0]:
            raise ValueError(
                f"Size mismatch: x has {x_data.shape[0]} rows, y has {y_data.shape[0]} rows."
            )
        if x_data.ndim != 2 or x_data.shape[1] < 1:
            raise ValueError(f"Expected x shape (n, d) with d >= 1, got {tuple(x_data.shape)}.")
        if y_data.ndim < 2 or y_data.shape[-1] != 1:
            raise ValueError(f"Expected y shape (n, ..., 1), got {tuple(y_data.shape)}.")
        if self.coll.ndim != 2 or self.coll.shape[1] < 1:
            raise ValueError(
                f"Expected coll shape (m, d) with d >= 1, got {tuple(self.coll.shape)}."
            )
        if x_data.shape[1] != self.coll.shape[1]:
            raise ValueError(
                f"Spatial dimension mismatch: x_data has d={x_data.shape[1]}, "
                f"coll has d={self.coll.shape[1]}. Both must share the same number of dimensions."
            )

        self._data_size = x_data.shape[0]

        self._context = InferredContext(
            x_data,
            y_data,
            self.validation,
        )

        self.pinn_ds = PINNDataset(
            x_data,
            y_data,
            self.coll,
            config.batch_size,
            config.data_ratio,
        )

        self.predict_ds = TensorDataset(
            x_data,
            y_data,
        )

        for callback in self.callbacks:
            callback.on_after_setup(self)

    @override
    def train_dataloader(self) -> DataLoader[TrainingBatch]:
        """
        Returns the training dataloader using PINNDataset.
        """
        return DataLoader[TrainingBatch](
            self.pinn_ds,
            batch_size=None,  # handled internally
            num_workers=cpu_count() or 1,
            persistent_workers=True,
            pin_memory=True,
        )

    @override
    def predict_dataloader(self) -> DataLoader[PredictionBatch]:
        """
        Returns the prediction dataloader using only the data dataset.
        """
        return DataLoader[PredictionBatch](
            cast(Dataset[PredictionBatch], self.predict_ds),
            batch_size=self._data_size,
            num_workers=cpu_count() or 1,
            persistent_workers=True,
            pin_memory=True,
        )

    @property
    def context(self) -> InferredContext:
        if self._context is None:
            raise RuntimeError("Context does not exist. Call setup() before accessing context.")
        return self._context
