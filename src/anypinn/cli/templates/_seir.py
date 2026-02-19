"""SEIR epidemic model template."""

from anypinn.cli._types import DataSource
from anypinn.cli.templates._base import train_py_core, train_py_lightning

EXPERIMENT_NAME = "seir-inverse"


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
"""SEIR epidemic model — mathematical definition."""

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
E_KEY = "E"
I_KEY = "I"
BETA_KEY = "beta"
SIGMA_KEY = "sigma"
GAMMA_KEY = "gamma"

# ============================================================================
# Constants
# ============================================================================

# True parameter values
TRUE_BETA = 0.5
TRUE_SIGMA = 1 / 5.2
TRUE_GAMMA = 1 / 10

# Initial conditions (fractions)
S0 = 0.99
E0 = 0.01
I0 = 0.001

# Time domain
T_DAYS = 160

# Noise level for synthetic data
NOISE_STD = 0.0005

# ============================================================================
# ODE Definition
# ============================================================================


def SEIR(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    """Scaled SEIR ODE system."""
    S, E, I = y
    b = args[BETA_KEY]
    sigma = args[SIGMA_KEY]
    gamma = args[GAMMA_KEY]

    dS = -b(x) * S * I
    dE = b(x) * S * I - sigma(x) * E
    dI = sigma(x) * E - gamma(x) * I

    dS = dS * T_DAYS
    dE = dE * T_DAYS
    dI = dI * T_DAYS
    return torch.stack([dS, dE, dI])


# ============================================================================
# Validation
# ============================================================================

{validation_block}

# ============================================================================
# Data Module Factory
# ============================================================================


def create_data_module(hp: ODEHyperparameters):
    from anypinn.catalog.seir import SEIRDataModule

    # Unscaled SEIR for data generation
    def SEIR_unscaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
        S, E, I = y
        b = args[BETA_KEY]
        sigma = args[SIGMA_KEY]
        gamma = args[GAMMA_KEY]
        dS = -b(x) * S * I
        dE = b(x) * S * I - sigma(x) * E
        dI = sigma(x) * E - gamma(x) * I
        return torch.stack([dS, dE, dI])

    gen_props = ODEProperties(
        ode=SEIR_unscaled,
        y0=torch.tensor([S0, E0, I0]),
        args={{
            BETA_KEY: Argument(TRUE_BETA),
            SIGMA_KEY: Argument(TRUE_SIGMA),
            GAMMA_KEY: Argument(TRUE_GAMMA),
        }},
    )

    return SEIRDataModule(
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
        ode=SEIR,
        y0=torch.tensor([S0, E0, I0]),
        args={{
            SIGMA_KEY: Argument(TRUE_SIGMA),
            GAMMA_KEY: Argument(TRUE_GAMMA),
        }},
    )

    fields = FieldsRegistry(
        {{
            S_KEY: Field(config=hp.fields_config),
            E_KEY: Field(config=hp.fields_config),
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
        y_columns=["I_obs"],  # TODO: update column names
    ),"""
    else:
        training_data = """\
    training_data=GenerationConfig(
        batch_size=100,
        data_ratio=2,
        collocations=6000,
        x=torch.linspace(start=0, end=160, steps=161),
        args_to_train={},
        noise_level=0,
    ),"""

    return f'''\
"""SEIR epidemic model — training configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from anypinn.core import (
    GenerationConfig,
    IngestionConfig,
    MLPConfig,
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
'''


def render(data_source: DataSource, lightning: bool) -> dict[str, str]:
    train_fn = train_py_lightning if lightning else train_py_core
    return {
        "ode.py": _ode_py(data_source),
        "config.py": _config_py(data_source),
        "train.py": train_fn(EXPERIMENT_NAME),
    }
