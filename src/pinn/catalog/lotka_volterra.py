from __future__ import annotations

from collections.abc import Sequence
from typing import override

import torch
from torch import Tensor
from torchdiffeq import odeint

from pinn.core import DataCallback, Domain1D, GenerationConfig, PINNDataModule, ValidationRegistry
from pinn.problems.ode import ODEHyperparameters, ODEProperties

X_KEY = "x"
Y_KEY = "y"
BETA_KEY = "beta"
ALPHA_KEY = "alpha"
DELTA_KEY = "delta"
GAMMA_KEY = "gamma"


class LotkaVolterraDataModule(PINNDataModule):
    """DataModule for Lotka-Volterra inverse problem. Generates synthetic data via odeint."""

    def __init__(
        self,
        hp: ODEHyperparameters,
        gen_props: ODEProperties,
        noise_frac: float = 0.0,
        validation: ValidationRegistry | None = None,
        callbacks: Sequence[DataCallback] | None = None,
    ):
        super().__init__(hp, validation, callbacks)
        self.gen_props = gen_props
        self.noise_frac = noise_frac

    @override
    def gen_coll(self, domain: Domain1D) -> Tensor:
        """Generate uniform collocation points."""
        coll = torch.rand((self.hp.training_data.collocations, 1))
        x0 = torch.tensor(domain.x0, dtype=torch.float32)
        x1 = torch.tensor(domain.x1, dtype=torch.float32)
        coll = coll * (x1 - x0) + x0
        return coll

    @override
    def gen_data(self, config: GenerationConfig) -> tuple[Tensor, Tensor]:
        """Generate synthetic Lotka-Volterra data using odeint + Gaussian noise."""

        def lv_ode(t: Tensor, y: Tensor) -> Tensor:
            return self.gen_props.ode(t, y, self.gen_props.args)

        t = config.x

        sol = odeint(lv_ode, self.gen_props.y0, t)  # [T, 2]
        x_true = sol[:, 0].clamp_min(0.0)
        y_true = sol[:, 1].clamp_min(0.0)

        x_obs = x_true + self.noise_frac * x_true.abs().mean() * torch.randn_like(x_true)
        y_obs = y_true + self.noise_frac * y_true.abs().mean() * torch.randn_like(y_true)
        x_obs = x_obs.clamp_min(0.0)
        y_obs = y_obs.clamp_min(0.0)

        y_data = torch.stack([x_obs, y_obs], dim=1).unsqueeze(-1)

        return t.unsqueeze(-1), y_data
