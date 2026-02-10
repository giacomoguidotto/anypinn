"""Blank project template — minimal skeleton."""

from anypinn.cli._types import DataSource
from anypinn.cli.templates._base import train_py_core, train_py_lightning

EXPERIMENT_NAME = "my-project"


def _ode_py(data_source: DataSource) -> str:
    return '''\
"""ODE mathematical definition."""

from __future__ import annotations

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
from anypinn.problems import ODEHyperparameters, ODEInverseProblem, ODEProperties

# TODO: define your ODE system here


validation: ValidationRegistry = {}


def create_data_module(hp: ODEHyperparameters):
    raise NotImplementedError("TODO: implement create_data_module")


def create_problem(hp: ODEHyperparameters) -> ODEInverseProblem:
    raise NotImplementedError("TODO: implement create_problem")
'''


def _config_py(data_source: DataSource) -> str:
    if data_source == DataSource.CSV:
        training_data = """\
    training_data=IngestionConfig(
        batch_size=100,
        data_ratio=2,
        collocations=6000,
        df_path=Path("./data/data.csv"),
        y_columns=["y_obs"],  # TODO: update column names
    ),"""
    else:
        training_data = """\
    training_data=GenerationConfig(
        batch_size=100,
        data_ratio=2,
        collocations=6000,
        x=torch.linspace(start=0, end=10, steps=100),
        args_to_train={},
        noise_level=0,
    ),"""

    return f'''\
"""Training configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from anypinn.core import (
    GenerationConfig,
    IngestionConfig,
    MLPConfig,
    ScalarConfig,
    SchedulerConfig,
)
from anypinn.problems import ODEHyperparameters


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
        init_value=0.5,
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
