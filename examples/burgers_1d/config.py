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

EXPERIMENT_NAME = "burgers-1d"

# ============================================================================
# Hyperparameters
# ============================================================================

hp = PINNHyperparameters(
    lr=1e-3,
    max_epochs=2000,
    gradient_clip_val=0.5,
    training_data=GenerationConfig(
        batch_size=128,
        data_ratio=2,
        collocations=10000,
        collocation_sampler="adaptive",
        x=torch.linspace(-1, 1, steps=50),
        args_to_train={},
        noise_level=0,
    ),
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
    scheduler=CosineAnnealingConfig(
        T_max=1000,
        eta_min=1e-6,
    ),
    smma_stopping=SMMAStoppingConfig(
        window=20,
        threshold=0.005,
        lookback=100,
    ),
)
