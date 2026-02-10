"""SIR epidemic model template."""

from anypinn.cli._types import DataSource
from anypinn.cli.templates._base import train_py_core, train_py_lightning

EXPERIMENT_NAME = "sir-inverse"


def _ode_py(data_source: DataSource) -> str:
    if data_source == DataSource.CSV:
        validation_block = """\
validation: ValidationRegistry = {
    # Map beta to a column in your CSV (e.g. reproduction number Rt * delta)
    BETA_KEY: ColumnRef(column="Rt", transform=lambda rt: rt * DELTA),
}"""
    else:
        validation_block = """\
validation: ValidationRegistry = {
    BETA_KEY: lambda x: torch.full_like(x, TRUE_BETA),
}"""

    return f'''\
"""SIR epidemic model — mathematical definition."""

from __future__ import annotations

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

S_KEY = "S"
I_KEY = "I"
BETA_KEY = "beta"
DELTA_KEY = "delta"
N_KEY = "N"

# ============================================================================
# Constants
# ============================================================================

# Scaling constants
C = 1e6  # population scale
T = 90   # time scale (days)

# Known parameters
N_POP = 56e6
DELTA = 1 / 5

# True parameter values (for synthetic data / validation)
TRUE_BETA = 0.6

# ============================================================================
# ODE Definition
# ============================================================================


def SIR(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    """Scaled SIR ODE system."""
    S, I = y
    b, d, N = args[BETA_KEY], args[DELTA_KEY], args[N_KEY]

    dS = -b(x) * I * S * C / N(x)
    dI = b(x) * I * S * C / N(x) - d(x) * I

    dS = dS * T
    dI = dI * T
    return torch.stack([dS, dI])


# ============================================================================
# Validation
# ============================================================================

{validation_block}

# ============================================================================
# Data Module Factory
# ============================================================================


def create_data_module(hp: ODEHyperparameters):
    from anypinn.catalog.sir import SIRInvDataModule

    return SIRInvDataModule(
        hp=hp,
        validation=validation,
        callbacks=[DataScaling(y_scale=1 / C)],
    )


# ============================================================================
# Problem Factory
# ============================================================================


def create_problem(hp: ODEHyperparameters) -> ODEInverseProblem:
    props = ODEProperties(
        ode=SIR,
        y0=torch.tensor([N_POP - 1, 1]) / C,
        args={{
            DELTA_KEY: Argument(DELTA),
            N_KEY: Argument(N_POP),
        }},
    )

    fields = FieldsRegistry(
        {{
            S_KEY: Field(config=hp.fields_config),
            I_KEY: Field(config=hp.fields_config),
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
        I_pred = fields[I_KEY](x_data)
        return cast(Tensor, I_pred)

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
        y_columns=["I_obs"],
    ),"""
    else:
        training_data = """\
    training_data=GenerationConfig(
        batch_size=100,
        data_ratio=2,
        collocations=6000,
        x=torch.linspace(start=0, end=90, steps=91),
        args_to_train={},
        noise_level=0,
    ),"""

    return f'''\
"""SIR epidemic model — training configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from anypinn.core import (
    GenerationConfig,
    IngestionConfig,
    MLPConfig,
    SchedulerConfig,
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
    lr=5e-4,
{training_data}
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
    scheduler=SchedulerConfig(
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
'''


def render(data_source: DataSource, lightning: bool) -> dict[str, str]:
    train_fn = train_py_lightning if lightning else train_py_core
    return {
        "ode.py": _ode_py(data_source),
        "config.py": _config_py(data_source),
        "train.py": train_fn(EXPERIMENT_NAME),
    }
