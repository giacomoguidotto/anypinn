"""Neural network primitives and building blocks for PINN."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast, override

import torch
from torch import Tensor
import torch.nn as nn

from anypinn.core.config import MLPConfig, ScalarConfig
from anypinn.core.types import Activations, Criteria


@dataclass
class Domain:
    """
    N-dimensional rectangular domain.

    Attributes:
        bounds: Per-dimension (min, max) pairs. ``bounds[i]`` covers dimension i.
        dx: Per-dimension step size (``None`` when not applicable).
    """

    bounds: list[tuple[float, float]]
    dx: list[float] | None = None

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return len(self.bounds)

    @property
    def x0(self) -> float:
        """Lower bound of the first dimension (convenience for 1-D / time-axis access)."""
        return self.bounds[0][0]

    @property
    def x1(self) -> float:
        """Upper bound of the first dimension."""
        return self.bounds[0][1]

    @classmethod
    def from_x(cls, x: Tensor) -> Domain:
        """
        Infer domain bounds and step sizes from a coordinate tensor of shape (N, d).

        Args:
            x: Coordinate tensor of shape ``(N, d)``.

        Returns:
            Domain with bounds and dx inferred from the data.
        """
        if x.ndim != 2:
            raise ValueError(f"Expected 2-D coordinate tensor (N, d), got shape {tuple(x.shape)}.")
        if x.shape[0] < 2:
            raise ValueError(
                f"At least two points are required to infer the domain, got {x.shape[0]}."
            )

        d = x.shape[1]
        bounds = [(x[:, i].min().item(), x[:, i].max().item()) for i in range(d)]
        dx = [(x[1, i] - x[0, i]).item() for i in range(d)]
        return cls(bounds=bounds, dx=dx)

    @override
    def __repr__(self) -> str:
        return f"Domain(ndim={self.ndim}, bounds={self.bounds}, dx={self.dx})"


def build_criterion(name: Criteria) -> nn.Module:
    """
    Return the loss-criterion module for the given name.

    Args:
        name: One of ``"mse"``, ``"huber"``, ``"l1"``.

    Returns:
        The corresponding PyTorch loss module.
    """
    return {
        "mse": nn.MSELoss(),
        "huber": nn.HuberLoss(),
        "l1": nn.L1Loss(),
    }[name]


def get_activation(name: Activations) -> nn.Module:
    """
    Get the activation function module by name.

    Args:
        name: The name of the activation function.

    Returns:
        The PyTorch activation module.
    """
    return {
        "tanh": nn.Tanh(),
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "sigmoid": nn.Sigmoid(),
        "selu": nn.SELU(),
        "softplus": nn.Softplus(),
        "identity": nn.Identity(),
    }[name]


class Field(nn.Module):
    """
    A neural field mapping coordinates -> vector of state variables.
    Example (ODE): t -> [S, I, R].

    Args:
        config: Configuration for the MLP backing this field.
    """

    def __init__(
        self,
        config: MLPConfig,
    ):
        super().__init__()
        encode = config.encode
        if isinstance(encode, nn.Module):
            # registers → participates in .to(), .state_dict()
            self.encoder: nn.Module | None = encode
        else:
            self.encoder = None
        self._encode_fn = encode  # callable reference (module or plain fn)
        dims = [config.in_dim] + config.hidden_layers + [config.out_dim]
        act = get_activation(config.activation)

        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act)

        if config.output_activation is not None:
            out_act = get_activation(config.output_activation)
            layers.append(out_act)

        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the field.

        Args:
            x: Input coordinates (e.g. time, space).

        Returns:
            The values of the field at input coordinates.
        """
        if self._encode_fn is not None:
            x = self._encode_fn(x)
        return cast(Tensor, self.net(x))


class Argument:
    """
    Represents an argument that can be passed to an ODE/PDE function.
    Can be a fixed float value or a callable function.

    Args:
        value: The value (float) or function (callable).
    """

    def __init__(self, value: float | Callable[[Tensor], Tensor]):
        self._value = value
        self._tensor_cache: dict[torch.device, Tensor] = {}

    def __call__(self, x: Tensor) -> Tensor:
        """
        Evaluate the argument.

        Args:
            x: Input tensor (context).

        Returns:
            The value of the argument, broadcasted if necessary.
        """
        if callable(self._value):
            return self._value(x)
        device = x.device
        if device not in self._tensor_cache:
            self._tensor_cache[device] = torch.tensor(self._value, device=device)
        return self._tensor_cache[device]

    @override
    def __repr__(self) -> str:
        return f"Argument(value={self._value})"


class Parameter(nn.Module, Argument):
    """
    Learnable parameter. Supports scalar or function-valued parameter.
    For function-valued parameters (e.g. β(t)), uses a small MLP.

    Args:
        config: Configuration for the parameter (ScalarConfig or MLPConfig).
    """

    def __init__(
        self,
        config: ScalarConfig | MLPConfig,
    ):
        super().__init__()
        self.config = config
        self._mode: Literal["scalar", "mlp"]

        if isinstance(config, ScalarConfig):
            self._mode = "scalar"
            self.value = nn.Parameter(torch.tensor(float(config.init_value), dtype=torch.float32))

        else:  # isinstance(config, MLPConfig)
            self._mode = "mlp"
            dims = [config.in_dim] + config.hidden_layers + [config.out_dim]
            act = get_activation(config.activation)

            layers: list[nn.Module] = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                if i < len(dims) - 2:
                    layers.append(act)

            if config.output_activation is not None:
                out_act = get_activation(config.output_activation)
                layers.append(out_act)

            self.net = nn.Sequential(*layers)
            self.apply(self._init)

    @property
    def mode(self) -> Literal["scalar", "mlp"]:
        """Mode of the parameter: 'scalar' or 'mlp'."""
        return self._mode

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    @override
    def forward(self, x: Tensor | None = None) -> Tensor:
        """
        Get the value of the parameter.

        Args:
            x: Input tensor (required for 'mlp' mode).

        Returns:
            The parameter value.
        """
        if self.mode == "scalar":
            return self.value if x is None else self.value.expand_as(x)
        else:
            if x is None:
                raise TypeError("Function-valued parameter requires input.")
            return cast(Tensor, self.net(x))


ArgsRegistry: TypeAlias = dict[str, Argument]
ParamsRegistry: TypeAlias = dict[str, Parameter]
FieldsRegistry: TypeAlias = dict[str, Field]
