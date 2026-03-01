"""Gray-Scott 2D Reaction-Diffusion — training configuration (synthetic)."""

from __future__ import annotations

import torch

from anypinn.core import (
    CosineAnnealingConfig,
    GenerationConfig,
    MLPConfig,
    ScalarConfig,
    SMMAStoppingConfig,
)
from anypinn.core.config import PINNHyperparameters

EXPERIMENT_NAME = "__EXPERIMENT_NAME__"
RUN_NAME = "v0"

# ============================================================================
# Hyperparameters
# ============================================================================

hp = PINNHyperparameters(
    lr=1e-3,
    max_epochs=3000,
    gradient_clip_val=0.5,
    training_data=GenerationConfig(
        batch_size=256,
        data_ratio=2,
        collocations=10000,
        collocation_sampler="latin_hypercube",
        x=torch.linspace(0, 1, steps=50),
        args_to_train={},
        noise_level=0,
    ),
    fields_config=MLPConfig(
        in_dim=39,  # FourierEncoding(K=6) on 3D: 3*(1+12) = 39
        out_dim=1,
        hidden_layers=[64, 128, 128, 64],
        activation="tanh",
        output_activation=None,
    ),
    params_config=ScalarConfig(
        init_value=0.01,  # matches D_u/D_v/F/k scale
    ),
    scheduler=CosineAnnealingConfig(
        T_max=1500,
        eta_min=1e-6,
    ),
    smma_stopping=SMMAStoppingConfig(
        window=20,
        threshold=0.005,
        lookback=100,
    ),
)
