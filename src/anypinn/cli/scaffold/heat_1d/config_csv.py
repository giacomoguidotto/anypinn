"""Heat Equation 1D — training configuration (CSV)."""

from __future__ import annotations

from pathlib import Path

from anypinn.core import IngestionConfig, MLPConfig, ReduceLROnPlateauConfig, ScalarConfig
from anypinn.core.config import PINNHyperparameters

EXPERIMENT_NAME = "__EXPERIMENT_NAME__"
RUN_NAME = "v0"

# ============================================================================
# Hyperparameters
# ============================================================================

hp = PINNHyperparameters(
    lr=1e-3,
    max_epochs=3000,
    gradient_clip_val=0.5,
    training_data=IngestionConfig(
        batch_size=128,
        data_ratio=2,
        collocations=8000,
        collocation_sampler="uniform",
        df_path=Path("./data/data.csv"),
        y_columns=["u"],  # TODO: update column names
    ),
    fields_config=MLPConfig(
        in_dim=26,  # FourierEncoding(K=6) on 2D: 2*(1+12) = 26
        out_dim=1,
        hidden_layers=[64, 128, 128, 64],
        activation="tanh",
        output_activation=None,
    ),
    params_config=ScalarConfig(
        init_value=0.5,  # initial guess for alpha (true=0.1)
    ),
    scheduler=ReduceLROnPlateauConfig(
        mode="min",
        factor=0.5,
        patience=100,
        threshold=1e-4,
        min_lr=1e-6,
    ),
)
