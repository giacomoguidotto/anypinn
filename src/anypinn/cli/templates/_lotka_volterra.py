"""Lotka-Volterra predator-prey template."""

from anypinn.cli._types import DataSource
from anypinn.cli.templates._base import train_py_core, train_py_lightning

EXPERIMENT_NAME = "lotka-volterra"


def _ode_py(data_source: DataSource) -> str:
    if data_source == DataSource.CSV:
        validation_block = """\
validation: ValidationRegistry = {
    # TODO: map beta to a column in your CSV
    # BETA_KEY: ColumnRef(column="your_column"),
}"""
    else:
        validation_block = """\
validation: ValidationRegistry = {
    BETA_KEY: lambda x: torch.full_like(x, TRUE_BETA),
}"""

    return f'''\
"""Lotka-Volterra predator-prey — mathematical definition."""

from __future__ import annotations

import torch
from torch import Tensor

from anypinn.core import (
    ArgsRegistry,
    Argument,
    ColumnRef,
    Field,
    FieldsRegistry,
    Parameter,
    ParamsRegistry,
    ValidationRegistry,
)
from anypinn.lightning.callbacks import DataScaling
from anypinn.problems import ODEHyperparameters, ODEInverseProblem, ODEProperties

# ============================================================================
# Keys
# ============================================================================

X_KEY = "x"
Y_KEY = "y"
ALPHA_KEY = "alpha"
BETA_KEY = "beta"
DELTA_KEY = "delta"
GAMMA_KEY = "gamma"

# ============================================================================
# Constants
# ============================================================================

# True parameter values
TRUE_ALPHA = 0.5
TRUE_BETA = 0.02
TRUE_DELTA = 0.01
TRUE_GAMMA = 0.5

# Initial conditions (populations)
X0 = 40.0
Y0 = 9.0

# Time domain
T_TOTAL = 50

# Population scaling
POP_SCALE = 100.0

# Noise level for synthetic data (std as fraction of signal)
NOISE_FRAC = 0.02

# ============================================================================
# Fourier encoding
# ============================================================================


def fourier_encode(t: Tensor) -> Tensor:
    features = [t]
    for k in range(1, 7):
        features.append(torch.sin(k * t))
    return torch.cat(features, dim=-1)


# ============================================================================
# ODE Definition
# ============================================================================


def lotka_volterra(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    """Scaled Lotka-Volterra ODE. Populations scaled by POP_SCALE, time by T_TOTAL."""
    prey, predator = y
    b = args[BETA_KEY]
    alpha = args[ALPHA_KEY]
    delta = args[DELTA_KEY]
    gamma = args[GAMMA_KEY]

    dx = alpha(x) * prey - b(x) * prey * predator * POP_SCALE
    dy = delta(x) * prey * predator * POP_SCALE - gamma(x) * predator

    dx = dx * T_TOTAL
    dy = dy * T_TOTAL
    return torch.stack([dx, dy])


# ============================================================================
# Validation
# ============================================================================

{validation_block}

# ============================================================================
# Data Module Factory
# ============================================================================


def create_data_module(hp: ODEHyperparameters):
    from anypinn.catalog.lotka_volterra import LotkaVolterraDataModule

    def LV_unscaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
        prey, predator = y
        b = args[BETA_KEY]
        alpha = args[ALPHA_KEY]
        delta = args[DELTA_KEY]
        gamma = args[GAMMA_KEY]
        dx = alpha(x) * prey - b(x) * prey * predator
        dy = delta(x) * prey * predator - gamma(x) * predator
        return torch.stack([dx, dy])

    gen_props = ODEProperties(
        ode=LV_unscaled,
        y0=torch.tensor([X0, Y0]),
        args={{
            ALPHA_KEY: Argument(TRUE_ALPHA),
            BETA_KEY: Argument(TRUE_BETA),
            DELTA_KEY: Argument(TRUE_DELTA),
            GAMMA_KEY: Argument(TRUE_GAMMA),
        }},
    )

    return LotkaVolterraDataModule(
        hp=hp,
        gen_props=gen_props,
        noise_frac=NOISE_FRAC,
        validation=validation,
        callbacks=[DataScaling(y_scale=[1 / POP_SCALE, 1 / POP_SCALE])],
    )


# ============================================================================
# Problem Factory
# ============================================================================


def create_problem(hp: ODEHyperparameters) -> ODEInverseProblem:
    props = ODEProperties(
        ode=lotka_volterra,
        y0=torch.tensor([X0, Y0]) / POP_SCALE,
        args={{
            ALPHA_KEY: Argument(TRUE_ALPHA),
            DELTA_KEY: Argument(TRUE_DELTA),
            GAMMA_KEY: Argument(TRUE_GAMMA),
        }},
    )

    fields = FieldsRegistry(
        {{
            X_KEY: Field(config=hp.fields_config),
            Y_KEY: Field(config=hp.fields_config),
        }}
    )
    params = ParamsRegistry(
        {{
            BETA_KEY: Parameter(config=hp.params_config),
        }}
    )

    def predict_data(
        x_data: Tensor, fields: FieldsRegistry, _params: ParamsRegistry
    ) -> Tensor:
        x_pred = fields[X_KEY](x_data)
        y_pred = fields[Y_KEY](x_data)
        return torch.stack([x_pred, y_pred])

    return ODEInverseProblem(
        props=props,
        hp=hp,
        fields=fields,
        params=params,
        predict_data=predict_data,
    )
'''


def _config_py(data_source: DataSource) -> str:
    if data_source == DataSource.CSV:
        training_data = """\
    training_data=IngestionConfig(
        batch_size=100,
        data_ratio=2,
        collocations=10000,
        df_path=Path("./data/data.csv"),
        y_columns=["x_obs", "y_obs"],  # TODO: update column names
    ),"""
    else:
        training_data = """\
    training_data=GenerationConfig(
        batch_size=100,
        data_ratio=2,
        collocations=10000,
        x=torch.linspace(start=0, end=50, steps=200),
        args_to_train={},
        noise_level=0,
    ),"""

    return f'''\
"""Lotka-Volterra predator-prey — training configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from anypinn.core import (
    GenerationConfig,
    IngestionConfig,
    MLPConfig,
    ScalarConfig,
    ReduceLROnPlateauConfig,
)
from anypinn.problems import ODEHyperparameters

from ode import fourier_encode

# ============================================================================
# Run Configuration
# ============================================================================


@dataclass
class RunConfig:
    max_epochs: int
    gradient_clip_val: float
    experiment_name: str
    run_name: str


CONFIG = RunConfig(
    max_epochs=5000,
    gradient_clip_val=1.0,
    experiment_name="{EXPERIMENT_NAME}",
    run_name="v0",
)

# ============================================================================
# Hyperparameters
# ============================================================================

hp = ODEHyperparameters(
    lr=1e-3,
{training_data}
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
'''


def render(data_source: DataSource, lightning: bool) -> dict[str, str]:
    train_fn = train_py_lightning if lightning else train_py_core
    return {
        "ode.py": _ode_py(data_source),
        "config.py": _config_py(data_source),
        "train.py": train_fn(EXPERIMENT_NAME),
    }
