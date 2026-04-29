"""Burgers Equation 1D — training configuration."""

from __future__ import annotations

from pathlib import Path

import torch

from anypinn.core import (
    GenerationConfig,
    IngestionConfig,
    MLPConfig,
    ReduceLROnPlateauConfig,
    ScalarConfig,
)
from anypinn.core.config import PINNHyperparameters

EXPERIMENT_NAME = "__EXPERIMENT_NAME__"
RUN_NAME = "v0"

# ============================================================================
# Hyperparameters
# ============================================================================

# --- VARIANT: source/synthetic ---
_training_data = GenerationConfig(
    batch_size=128,
    data_ratio=2,
    collocations=10000,
    collocation_sampler="adaptive",
    x=torch.linspace(-1, 1, steps=50),
    args_to_train={},
    noise_level=0,
)
# --- VARIANT: source/csv ---
_training_data = IngestionConfig(
    batch_size=128,
    data_ratio=2,
    collocations=10000,
    collocation_sampler="adaptive",
    df_path=Path("./data/data.csv"),
    y_columns=["u"],  # TODO: update column names
)
# --- END VARIANT ---

hp = PINNHyperparameters(
    lr=1e-3,
    max_epochs=5000,
    gradient_clip_val=0.5,
    training_data=_training_data,
    fields_config=MLPConfig(
        in_dim=256,  # RandomFourierFeatures(num_features=128) -> out_dim=256
        out_dim=1,
        hidden_layers=[64, 128, 128, 64],
        activation="tanh",
        output_activation=None,
    ),
    params_config=ScalarConfig(
        init_value=0.1,  # initial guess for nu (true=0.01/pi)
    ),
    scheduler=ReduceLROnPlateauConfig(
        mode="min",
        factor=0.5,
        patience=200,
        threshold=1e-4,
        min_lr=1e-6,
    ),
)
