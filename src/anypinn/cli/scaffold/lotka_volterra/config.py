"""Lotka-Volterra predator-prey — training configuration."""

from __future__ import annotations

from pathlib import Path

from ode import fourier_encode
import torch

from anypinn.core import (
    GenerationConfig,
    IngestionConfig,
    MLPConfig,
    ReduceLROnPlateauConfig,
    ScalarConfig,
)
from anypinn.problems import ODEHyperparameters

EXPERIMENT_NAME = "__EXPERIMENT_NAME__"
RUN_NAME = "v0"

# ============================================================================
# Hyperparameters
# ============================================================================

# --- VARIANT: source/synthetic ---
_training_data = GenerationConfig(
    batch_size=100,
    data_ratio=2,
    collocations=10000,
    x=torch.linspace(start=0, end=50, steps=200),
    args_to_train={},
    noise_level=0,
)
# --- VARIANT: source/csv ---
_training_data = IngestionConfig(
    batch_size=100,
    data_ratio=2,
    collocations=10000,
    df_path=Path("./data/data.csv"),
    y_columns=["x_obs", "y_obs"],  # TODO: update column names
)
# --- END VARIANT ---

hp = ODEHyperparameters(
    lr=1e-3,
    max_epochs=5000,
    gradient_clip_val=1.0,
    training_data=_training_data,
    fields_config=MLPConfig(
        in_dim=7,
        out_dim=1,
        hidden_layers=[64, 64, 64, 64, 64, 64],
        activation="tanh",
        output_activation=None,
        encode=fourier_encode,
    ),
    params_config=ScalarConfig(
        init_value=0.05,
    ),
    scheduler=ReduceLROnPlateauConfig(
        mode="min",
        factor=0.5,
        patience=200,
        threshold=1e-4,
        min_lr=1e-5,
    ),
    pde_weight=1.0,
    ic_weight=10,
    data_weight=1.0,
)
