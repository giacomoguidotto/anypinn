from __future__ import annotations

from dataclasses import dataclass

from ode import T_TOTAL
import torch

from anypinn.core import GenerationConfig, MLPConfig, ReduceLROnPlateauConfig, ScalarConfig
from anypinn.problems import ODEHyperparameters


@dataclass
class RunConfig:
    experiment_name: str


CONFIG = RunConfig(
    experiment_name="van-der-pol",
)

hp = ODEHyperparameters(
    lr=1e-3,
    max_epochs=3000,
    gradient_clip_val=0.1,
    training_data=GenerationConfig(
        batch_size=100,
        data_ratio=2,
        collocations=8000,
        x=torch.linspace(start=0, end=T_TOTAL, steps=400),
        noise_level=0,
        args_to_train={},
    ),
    fields_config=MLPConfig(
        in_dim=1,
        out_dim=1,
        hidden_layers=[64, 128, 128, 64],
        activation="tanh",
        output_activation=None,
    ),
    params_config=ScalarConfig(
        init_value=2.0,
    ),
    scheduler=ReduceLROnPlateauConfig(
        mode="min",
        factor=0.5,
        patience=55,
        threshold=5e-3,
        min_lr=1e-6,
    ),
    pde_weight=1e-4,
    ic_weight=15,
    data_weight=5,
)
