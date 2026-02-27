from __future__ import annotations

from collections.abc import Sequence
from typing import override

import torch
from torch import Tensor
from torchdiffeq import odeint

from anypinn.core import DataCallback, GenerationConfig, PINNDataModule, ValidationRegistry
from anypinn.problems.ode import ODEHyperparameters, ODEProperties

U_KEY = "u"
MU_KEY = "mu"


class VanDerPolDataModule(PINNDataModule):
    """DataModule for Van der Pol oscillator inverse problem.

    Generates synthetic data via odeint.
    """

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
        """Generate synthetic Van der Pol data using odeint + Gaussian noise."""

        def vdp_ode(t: Tensor, y: Tensor) -> Tensor:
            return self.gen_props.ode(t, y, self.gen_props.args)

        t = config.x

        sol = odeint(vdp_ode, self.gen_props.y0, t)  # [T, 2]
        u_true = sol[:, 0]

        u_obs = u_true + self.noise_std * torch.randn_like(u_true)

        return t.unsqueeze(-1), u_obs.unsqueeze(-1).unsqueeze(1)
