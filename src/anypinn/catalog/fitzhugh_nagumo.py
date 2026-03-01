from __future__ import annotations

from collections.abc import Sequence
from typing import override

import torch
from torch import Tensor
from torchdiffeq import odeint

from anypinn.core import DataCallback, GenerationConfig, PINNDataModule, ValidationRegistry
from anypinn.problems.ode import ODEHyperparameters, ODEProperties

V_KEY = "v"
W_KEY = "w"
EPSILON_KEY = "epsilon"
A_KEY = "a"


class FitzHughNagumoDataModule(PINNDataModule):
    """DataModule for FitzHugh-Nagumo inverse problem.

    Generates synthetic data via odeint. Only the voltage v is observed;
    the recovery variable w is inferred through ODE residuals alone.
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
        """Generate synthetic FHN data. Returns only v (partially observed)."""

        def fhn_ode(t: Tensor, y: Tensor) -> Tensor:
            return self.gen_props.ode(t, y, self.gen_props.args)

        t = config.x

        sol = odeint(fhn_ode, self.gen_props.y0, t)  # [T, 2]
        v_true = sol[:, 0]

        v_obs = v_true + self.noise_std * torch.randn_like(v_true)

        # (N, 1, 1) — single observed field
        y_data = v_obs.unsqueeze(1).unsqueeze(-1)

        return t.unsqueeze(-1), y_data
