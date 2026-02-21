"""Shared fixtures for the PINN test suite."""

import pytest
import torch
from torch import Tensor

from anypinn.core.config import GenerationConfig, MLPConfig, ScalarConfig
from anypinn.core.context import InferredContext
from anypinn.core.nn import Argument, Domain, Field, Parameter
from anypinn.core.types import TrainingBatch
from anypinn.core.validation import ResolvedValidation


@pytest.fixture
def simple_mlp_config() -> MLPConfig:
    return MLPConfig(
        in_dim=1,
        out_dim=1,
        hidden_layers=[8, 8],
        activation="tanh",
    )


@pytest.fixture
def multi_output_mlp_config() -> MLPConfig:
    return MLPConfig(
        in_dim=1,
        out_dim=3,
        hidden_layers=[8, 8],
        activation="tanh",
    )


@pytest.fixture
def scalar_config() -> ScalarConfig:
    return ScalarConfig(init_value=0.5)


@pytest.fixture
def field(simple_mlp_config: MLPConfig) -> Field:
    return Field(simple_mlp_config)


@pytest.fixture
def multi_field(multi_output_mlp_config: MLPConfig) -> Field:
    return Field(multi_output_mlp_config)


@pytest.fixture
def scalar_param(scalar_config: ScalarConfig) -> Parameter:
    return Parameter(scalar_config)


@pytest.fixture
def mlp_param(simple_mlp_config: MLPConfig) -> Parameter:
    return Parameter(simple_mlp_config)


@pytest.fixture
def x_tensor() -> Tensor:
    """Sample x data: 50 evenly spaced points in [0, 10]."""
    return torch.linspace(0, 10, 50).unsqueeze(-1)


@pytest.fixture
def y_tensor() -> Tensor:
    """Sample y data: 50 points, single output."""
    return torch.randn(50, 1)


@pytest.fixture
def training_batch(x_tensor: Tensor, y_tensor: Tensor) -> TrainingBatch:
    """A training batch: ((x_data, y_data), x_coll)."""
    x_coll = torch.rand(100, 1) * 10
    return ((x_tensor[:10], y_tensor[:10]), x_coll)


@pytest.fixture
def domain() -> Domain:
    return Domain(bounds=[(0.0, 10.0)], dx=[0.2])


@pytest.fixture
def resolved_validation() -> ResolvedValidation:
    return {}


@pytest.fixture
def context(
    x_tensor: Tensor, y_tensor: Tensor, resolved_validation: ResolvedValidation
) -> InferredContext:
    return InferredContext(x_tensor, y_tensor, resolved_validation)


def simple_ode(x: Tensor, y: Tensor, args: dict[str, Argument]) -> Tensor:
    """dy/dt = -y (exponential decay)."""
    return -y


@pytest.fixture
def generation_config() -> GenerationConfig:
    return GenerationConfig(
        batch_size=32,
        data_ratio=0.5,
        collocations=100,
        x=torch.linspace(0, 10, 50),
        noise_level=0.0,
        args_to_train={},
    )
