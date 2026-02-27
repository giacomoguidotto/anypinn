"""Lorenz system — training configuration."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from anypinn.core import GenerationConfig, MLPConfig, ReduceLROnPlateauConfig, ScalarConfig
from anypinn.problems import ODEHyperparameters

# ============================================================================
# Run Configuration
# ============================================================================


@dataclass
class RunConfig:
    experiment_name: str
    run_name: str


CONFIG = RunConfig(
    experiment_name="__EXPERIMENT_NAME__",
    run_name="v0",
)

# ============================================================================
# Hyperparameters
# ============================================================================

hp = ODEHyperparameters(
    lr=1e-3,
    max_epochs=3000,
    gradient_clip_val=0.5,
    criterion="huber",
    training_data=GenerationConfig(
        batch_size=100,
        data_ratio=2,
        collocations=8000,
        collocation_sampler="latin_hypercube",
        x=torch.linspace(start=0, end=3, steps=300),
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
        init_value=15.0,
    ),
    scheduler=ReduceLROnPlateauConfig(
        mode="min",
        factor=0.5,
        patience=55,
        threshold=5e-3,
        min_lr=1e-6,
    ),
    pde_weight=1e-3,
    ic_weight=10,
    data_weight=5,
)
