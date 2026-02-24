"""Training configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from anypinn.core import (
    GenerationConfig,
    IngestionConfig,
    MLPConfig,
    ScalarConfig,
    ReduceLROnPlateauConfig,
)
from anypinn.problems import ODEHyperparameters


@dataclass
class RunConfig:
    max_epochs: int
    gradient_clip_val: float
    experiment_name: str
    run_name: str


CONFIG = RunConfig(
    max_epochs=2000,
    gradient_clip_val=0.1,
    experiment_name="__EXPERIMENT_NAME__",
    run_name="v0",
)

hp = ODEHyperparameters(
    lr=1e-3,
    training_data=IngestionConfig(
        batch_size=100,
        data_ratio=2,
        collocations=6000,
        df_path=Path("./data/data.csv"),
        y_columns=["y_obs"],  # TODO: update column names
    ),
    fields_config=MLPConfig(
        in_dim=1,
        out_dim=1,
        hidden_layers=[64, 128, 128, 64],
        activation="tanh",
        output_activation=None,
    ),
    params_config=ScalarConfig(
        init_value=0.5,
    ),
    pde_weight=1,
    ic_weight=1,
    data_weight=1,
)
