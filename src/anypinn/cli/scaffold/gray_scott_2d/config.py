"""Gray-Scott 2D Reaction-Diffusion — training configuration."""

from __future__ import annotations

from pathlib import Path

import torch

from anypinn.core import (
    CosineAnnealingConfig,
    GenerationConfig,
    IngestionConfig,
    MLPConfig,
    ReduceLROnPlateauConfig,
    ScalarConfig,
    SMMAStoppingConfig,
)
from anypinn.core.config import PINNHyperparameters

EXPERIMENT_NAME = "__EXPERIMENT_NAME__"
RUN_NAME = "v0"

# ============================================================================
# Hyperparameters
# ============================================================================

# --- VARIANT: source/synthetic ---
_training_data = GenerationConfig(
    batch_size=256,
    data_ratio=64,
    collocations=10000,
    collocation_sampler="latin_hypercube",
    x=torch.linspace(0, 1, steps=50),
    args_to_train={},
    noise_level=0,
)
# --- VARIANT: source/csv ---
_training_data = IngestionConfig(
    batch_size=256,
    data_ratio=64,
    collocations=10000,
    collocation_sampler="latin_hypercube",
    df_path=Path("./data/data.csv"),
    y_columns=["u", "v"],  # TODO: update column names
)
# --- END VARIANT ---

# --- VARIANT: source/synthetic ---
hp = PINNHyperparameters(
    lr=1e-3,
    max_epochs=3000,
    gradient_clip_val=0.5,
    training_data=_training_data,
    fields_config=MLPConfig(
        in_dim=39,  # FourierEncoding(K=6) on 3D: 3*(1+12) = 39
        out_dim=1,
        hidden_layers=[128, 256, 256, 128],
        activation="tanh",
        output_activation=None,
    ),
    params_config=ScalarConfig(
        init_value=0.01,
    ),
    scheduler=CosineAnnealingConfig(
        T_max=3000,
        eta_min=1e-6,
    ),
    smma_stopping=SMMAStoppingConfig(
        window=20,
        threshold=0.005,
        lookback=100,
    ),
)
# --- VARIANT: source/csv ---
hp = PINNHyperparameters(
    lr=1e-3,
    max_epochs=3000,
    gradient_clip_val=0.5,
    training_data=_training_data,
    fields_config=MLPConfig(
        in_dim=39,  # FourierEncoding(K=6) on 3D: 3*(1+12) = 39
        out_dim=1,
        hidden_layers=[128, 256, 256, 128],
        activation="tanh",
        output_activation=None,
    ),
    params_config=ScalarConfig(
        init_value=0.01,  # matches D_u/D_v/F/k scale
    ),
    scheduler=ReduceLROnPlateauConfig(
        mode="min",
        factor=0.5,
        patience=100,
        threshold=1e-4,
        min_lr=1e-6,
    ),
)
# --- END VARIANT ---
