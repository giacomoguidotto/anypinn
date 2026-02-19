"""Damped oscillator template."""

from anypinn.cli._types import DataSource
from anypinn.cli.templates._base import train_py_core, train_py_lightning

EXPERIMENT_NAME = "damped-oscillator"


def _ode_py(data_source: DataSource) -> str:
    if data_source == DataSource.CSV:
        validation_block = """\
validation: ValidationRegistry = {
    # TODO: map zeta to a column in your CSV
    # ZETA_KEY: ColumnRef(column="your_column"),
}"""
    else:
        validation_block = """\
validation: ValidationRegistry = {
    ZETA_KEY: lambda x: torch.full_like(x, TRUE_ZETA),
}"""

    return f'''\
"""Damped oscillator — mathematical definition."""

from __future__ import annotations

import math
from typing import cast

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
V_KEY = "v"
ZETA_KEY = "zeta"
OMEGA_KEY = "omega0"

# ============================================================================
# Constants
# ============================================================================

# True parameter values
TRUE_ZETA = 0.15
TRUE_OMEGA0 = 2 * math.pi

# Initial conditions
X0 = 1.0
V0 = 0.0

# Time domain (seconds)
T_TOTAL = 5

# Noise level for synthetic data
NOISE_STD = 0.02

# ============================================================================
# ODE Definition
# ============================================================================


def oscillator(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    """Scaled damped oscillator ODE: dx/dt = v, dv/dt = -2*zeta*omega0*v - omega0^2*x."""
    pos, vel = y
    z = args[ZETA_KEY]
    omega0 = args[OMEGA_KEY]

    dx = vel
    dv = -2 * z(x) * omega0(x) * vel - omega0(x) ** 2 * pos

    dx = dx * T_TOTAL
    dv = dv * T_TOTAL
    return torch.stack([dx, dv])


# ============================================================================
# Validation
# ============================================================================

{validation_block}

# ============================================================================
# Data Module Factory
# ============================================================================


def create_data_module(hp: ODEHyperparameters):
    from anypinn.catalog.damped_oscillator import DampedOscillatorDataModule

    def oscillator_unscaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
        pos, vel = y
        z = args[ZETA_KEY]
        omega0 = args[OMEGA_KEY]
        dx = vel
        dv = -2 * z(x) * omega0(x) * vel - omega0(x) ** 2 * pos
        return torch.stack([dx, dv])

    gen_props = ODEProperties(
        ode=oscillator_unscaled,
        y0=torch.tensor([X0, V0]),
        args={{
            ZETA_KEY: Argument(TRUE_ZETA),
            OMEGA_KEY: Argument(TRUE_OMEGA0),
        }},
    )

    return DampedOscillatorDataModule(
        hp=hp,
        gen_props=gen_props,
        noise_std=NOISE_STD,
        validation=validation,
        callbacks=[DataScaling(y_scale=1.0)],
    )


# ============================================================================
# Problem Factory
# ============================================================================


def create_problem(hp: ODEHyperparameters) -> ODEInverseProblem:
    props = ODEProperties(
        ode=oscillator,
        y0=torch.tensor([X0, V0]),
        args={{
            OMEGA_KEY: Argument(TRUE_OMEGA0),
        }},
    )

    fields = FieldsRegistry(
        {{
            X_KEY: Field(config=hp.fields_config),
            V_KEY: Field(config=hp.fields_config),
        }}
    )
    params = ParamsRegistry(
        {{
            ZETA_KEY: Parameter(config=hp.params_config),
        }}
    )

    def predict_data(
        x_data: Tensor, fields: FieldsRegistry, _params: ParamsRegistry
    ) -> Tensor:
        x_pred = fields[X_KEY](x_data)
        return cast(Tensor, x_pred)

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
        collocations=6000,
        df_path=Path("./data/data.csv"),
        y_columns=["x_obs"],  # TODO: update column names
    ),"""
    else:
        training_data = """\
    training_data=GenerationConfig(
        batch_size=100,
        data_ratio=2,
        collocations=6000,
        x=torch.linspace(start=0, end=5, steps=200),
        args_to_train={},
        noise_level=0,
    ),"""

    return f'''\
"""Damped oscillator — training configuration."""

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
    max_epochs=2000,
    gradient_clip_val=0.1,
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
        in_dim=1,
        out_dim=1,
        hidden_layers=[64, 128, 128, 64],
        activation="tanh",
        output_activation=None,
    ),
    params_config=ScalarConfig(
        init_value=0.3,
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
'''


def render(data_source: DataSource, lightning: bool) -> dict[str, str]:
    train_fn = train_py_lightning if lightning else train_py_core
    return {
        "ode.py": _ode_py(data_source),
        "config.py": _config_py(data_source),
        "train.py": train_fn(EXPERIMENT_NAME),
    }
