from __future__ import annotations

from collections.abc import Sequence
from typing import override

import torch
from torch import Tensor
from torchdiffeq import odeint

from anypinn.core import DataCallback, GenerationConfig, PINNDataModule, ValidationRegistry
from anypinn.problems.ode import ODEHyperparameters, ODEProperties

X_KEY = "x"
Y_KEY = "y"
Z_KEY = "z"
SIGMA_KEY = "sigma"
RHO_KEY = "rho"
BETA_KEY = "beta"


class LorenzDataModule(PINNDataModule):
    """DataModule for Lorenz system inverse problem. Generates synthetic data via odeint."""

    def __init__(
        self,
        hp: ODEHyperparameters,
        gen_props: ODEProperties,
        noise_std: float = 0.0,
        validation: ValidationRegistry | None = None,
        callbacks: Sequence[DataCallback] | None = None,
    ):
        super().__init__(hp, validation, callbacks)
        self.gen_props = gen_props
        self.noise_std = noise_std

    @override
    def gen_data(self, config: GenerationConfig) -> tuple[Tensor, Tensor]:
        """Generate synthetic Lorenz data using odeint + additive Gaussian noise."""

        def lorenz_ode(t: Tensor, y: Tensor) -> Tensor:
            return self.gen_props.ode(t, y, self.gen_props.args)

        t = config.x

        sol = odeint(lorenz_ode, self.gen_props.y0, t)  # [T, 3]
        x_true = sol[:, 0]
        y_true = sol[:, 1]
        z_true = sol[:, 2]

        x_obs = x_true + self.noise_std * torch.randn_like(x_true)
        y_obs = y_true + self.noise_std * torch.randn_like(y_true)
        z_obs = z_true + self.noise_std * torch.randn_like(z_true)

        y_data = torch.stack([x_obs, y_obs, z_obs], dim=1).unsqueeze(-1)

        return t.unsqueeze(-1), y_data
