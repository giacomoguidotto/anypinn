from __future__ import annotations

from collections.abc import Sequence
from typing import override

import torch
from torch import Tensor
from torchdiffeq import odeint

from anypinn.core import DataCallback, GenerationConfig, PINNDataModule, ValidationRegistry
from anypinn.problems.ode import ODEHyperparameters, ODEProperties

S_KEY = "S"
E_KEY = "E"
I_KEY = "I"
BETA_KEY = "beta"
SIGMA_KEY = "sigma"
GAMMA_KEY = "gamma"


class SEIRDataModule(PINNDataModule):
    """DataModule for SEIR inverse problem. Generates synthetic data via odeint."""

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
        """Generate synthetic SEIR data using odeint + Gaussian noise."""

        def seir_ode(t: Tensor, y: Tensor) -> Tensor:
            return self.gen_props.ode(t, y, self.gen_props.args)

        t = config.x

        sol = odeint(seir_ode, self.gen_props.y0, t)  # [T, 3]
        I_true = sol[:, 2].clamp_min(0.0)

        I_obs = I_true + self.noise_std * torch.randn_like(I_true)
        I_obs = I_obs.clamp_min(0.0)

        return t.unsqueeze(-1), I_obs.unsqueeze(-1).unsqueeze(1)
