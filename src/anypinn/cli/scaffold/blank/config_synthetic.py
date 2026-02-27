"""Training configuration."""

from __future__ import annotations

import torch

from anypinn.core import GenerationConfig, MLPConfig, ScalarConfig
from anypinn.problems import ODEHyperparameters

EXPERIMENT_NAME = "__EXPERIMENT_NAME__"
RUN_NAME = "v0"

hp = ODEHyperparameters(
    lr=1e-3,
    max_epochs=2000,
    gradient_clip_val=0.1,
    training_data=GenerationConfig(
        batch_size=100,
        data_ratio=2,
        collocations=6000,
        x=torch.linspace(start=0, end=10, steps=100),
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
        init_value=0.5,
    ),
    pde_weight=1,
    ic_weight=1,
    data_weight=1,
)
