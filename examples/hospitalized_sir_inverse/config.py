from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from anypinn.core import IngestionConfig, MLPConfig, ReduceLROnPlateauConfig
from anypinn.problems import ODEHyperparameters


@dataclass
class RunConfig:
    max_epochs: int
    gradient_clip_val: float
    experiment_name: str


CONFIG = RunConfig(
    max_epochs=2000,
    gradient_clip_val=0.1,
    experiment_name="hospitalized-sir-inverse",
)

hp = ODEHyperparameters(
    lr=5e-4,
    training_data=IngestionConfig(
        batch_size=100,
        data_ratio=2,
        collocations=6000,
        df_path=Path("./data/synt_h_data.csv"),
        y_columns=["I_obs", "H_obs"],
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
)
