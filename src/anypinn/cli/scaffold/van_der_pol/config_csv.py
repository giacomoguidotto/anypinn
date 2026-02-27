"""Van der Pol oscillator — training configuration."""

from __future__ import annotations

from pathlib import Path

from anypinn.core import IngestionConfig, MLPConfig, ReduceLROnPlateauConfig, ScalarConfig
from anypinn.problems import ODEHyperparameters

EXPERIMENT_NAME = "__EXPERIMENT_NAME__"
RUN_NAME = "v0"

# ============================================================================
# Hyperparameters
# ============================================================================

hp = ODEHyperparameters(
    lr=1e-3,
    max_epochs=3000,
    gradient_clip_val=0.1,
    training_data=IngestionConfig(
        batch_size=100,
        data_ratio=2,
        collocations=8000,
        df_path=Path("./data/data.csv"),
        y_columns=["u_obs"],  # TODO: update column names
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
