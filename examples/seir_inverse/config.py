from __future__ import annotations

from dataclasses import dataclass

from ode import T_DAYS
import torch

from anypinn.core import GenerationConfig, MLPConfig, ReduceLROnPlateauConfig
from anypinn.problems import ODEHyperparameters


@dataclass
class RunConfig:
    experiment_name: str


CONFIG = RunConfig(
    experiment_name="seir-inverse",
)

hp = ODEHyperparameters(
    lr=5e-4,
    max_epochs=2000,
    gradient_clip_val=0.1,
    training_data=GenerationConfig(
        batch_size=100,
        data_ratio=2,
        collocations=6000,
        x=torch.linspace(start=0, end=T_DAYS, steps=T_DAYS + 1),
        noise_level=0,
        args_to_train={},
    ),
    fields_config=MLPConfig(
        in_dim=1,
        out_dim=1,
        hidden_layers=[64, 128, 128, 64],
        activation="tanh",
        output_activation="softplus",
    ),
    params_config=MLPConfig(
        in_dim=1,
        out_dim=1,
        hidden_layers=[64, 128, 128, 64],
        activation="tanh",
        output_activation="softplus",
    ),
    scheduler=ReduceLROnPlateauConfig(
        mode="min",
        factor=0.5,
        patience=55,
        threshold=5e-3,
        min_lr=1e-6,
    ),
    pde_weight=1,
    ic_weight=1,
    data_weight=1,
)
