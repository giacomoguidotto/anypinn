"""SIR epidemic model — training configuration."""

from __future__ import annotations

from pathlib import Path

from anypinn.core import IngestionConfig, MLPConfig, ReduceLROnPlateauConfig
from anypinn.problems import ODEHyperparameters

EXPERIMENT_NAME = "__EXPERIMENT_NAME__"
RUN_NAME = "v0"

# ============================================================================
# Hyperparameters
# ============================================================================

hp = ODEHyperparameters(
    lr=5e-4,
    max_epochs=2000,
    gradient_clip_val=0.1,
    training_data=IngestionConfig(
        batch_size=100,
        data_ratio=2,
        collocations=6000,
        df_path=Path("./data/data.csv"),
        y_columns=["I_obs"],
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
