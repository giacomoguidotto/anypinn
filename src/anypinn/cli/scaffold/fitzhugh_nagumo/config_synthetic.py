"""FitzHugh-Nagumo neuron model — training configuration."""

from __future__ import annotations

import torch

from anypinn.core import GenerationConfig, LBFGSConfig, MLPConfig, ScalarConfig, SMMAStoppingConfig
from anypinn.problems import ODEHyperparameters

EXPERIMENT_NAME = "__EXPERIMENT_NAME__"
RUN_NAME = "v0"

# ============================================================================
# Hyperparameters
# ============================================================================

hp = ODEHyperparameters(
    lr=1.0,
    max_epochs=500,
    criterion="mse",
    optimizer=LBFGSConfig(lr=1.0, max_iter=20, history_size=100),
    training_data=GenerationConfig(
        batch_size=300,
        data_ratio=2,
        collocations=5000,
        collocation_sampler="latin_hypercube",
        x=torch.linspace(start=0, end=50, steps=300),
        args_to_train={},
        noise_level=0,
    ),
    fields_config=MLPConfig(
        in_dim=1,
        out_dim=1,
        hidden_layers=[64, 128, 128, 64],
        activation="tanh",
        output_activation=None,
    ),
    params_config=ScalarConfig(
        init_value=0.3,
    ),
    smma_stopping=SMMAStoppingConfig(
        window=50,
        threshold=1e-6,
        lookback=100,
    ),
    pde_weight=1e-3,
    ic_weight=10,
    data_weight=5,
)
