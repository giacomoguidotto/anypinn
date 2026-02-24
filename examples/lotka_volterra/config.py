from __future__ import annotations

from dataclasses import dataclass

from ode import T_TOTAL, fourier_encode
import torch

from anypinn.core import GenerationConfig, MLPConfig, ReduceLROnPlateauConfig, ScalarConfig
from anypinn.problems import ODEHyperparameters


@dataclass
class RunConfig:
    max_epochs: int
    gradient_clip_val: float
    experiment_name: str


CONFIG = RunConfig(
    max_epochs=5000,
    gradient_clip_val=1.0,
    experiment_name="lotka-volterra",
)

hp = ODEHyperparameters(
    lr=1e-3,
    training_data=GenerationConfig(
        batch_size=100,
        data_ratio=2,
        collocations=10000,
        x=torch.linspace(start=0, end=T_TOTAL, steps=200),
        noise_level=0,
        args_to_train={},
    ),
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
