from __future__ import annotations

import torch

from anypinn.core import (
    GenerationConfig,
    MLPConfig,
    ReduceLROnPlateauConfig,
    ScalarConfig,
    SMMAStoppingConfig,
)
from anypinn.core.config import PINNHyperparameters

EXPERIMENT_NAME = "inverse-diffusivity"

# ============================================================================
# Hyperparameters
# ============================================================================

hp = PINNHyperparameters(
    lr=1e-3,
    max_epochs=3000,
    gradient_clip_val=0.5,
    training_data=GenerationConfig(
        batch_size=128,
        data_ratio=2,
        collocations=8000,
        collocation_sampler="uniform",
        x=torch.linspace(0, 1, steps=50),
        args_to_train={},
        noise_level=0,
    ),
    fields_config=MLPConfig(
        in_dim=26,  # FourierEncoding(K=6) on 2D: 2*(1+12) = 26
        out_dim=1,
        hidden_layers=[64, 128, 128, 64],
        activation="tanh",
        output_activation=None,
    ),
    params_config=ScalarConfig(
        init_value=0.0,  # placeholder, unused (no scalar params)
    ),
    scheduler=ReduceLROnPlateauConfig(
        mode="min",
        factor=0.5,
        patience=100,
        threshold=1e-4,
        min_lr=1e-6,
    ),
    smma_stopping=SMMAStoppingConfig(
        window=20,
        threshold=0.005,
        lookback=100,
    ),
)
