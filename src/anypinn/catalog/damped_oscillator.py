from __future__ import annotations

from collections.abc import Sequence
from typing import override

import torch
from torch import Tensor
from torchdiffeq import odeint

from anypinn.core import DataCallback, GenerationConfig, PINNDataModule, ValidationRegistry
from anypinn.problems.ode import ODEHyperparameters, ODEProperties

X_KEY = "x"
V_KEY = "v"
ZETA_KEY = "zeta"
OMEGA_KEY = "omega0"


class DampedOscillatorDataModule(PINNDataModule):
    """DataModule for damped oscillator inverse problem. Generates synthetic data via odeint."""

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
        """Generate synthetic damped oscillator data using odeint + Gaussian noise."""

        def oscillator_ode(t: Tensor, y: Tensor) -> Tensor:
            return self.gen_props.ode(t, y, self.gen_props.args)

        t = config.x

        sol = odeint(oscillator_ode, self.gen_props.y0, t)  # [T, 2]
        x_true = sol[:, 0]

        x_obs = x_true + self.noise_std * torch.randn_like(x_true)

        return t.unsqueeze(-1), x_obs.unsqueeze(-1).unsqueeze(1)
