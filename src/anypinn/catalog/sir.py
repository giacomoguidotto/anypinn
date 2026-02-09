from __future__ import annotations

from collections.abc import Sequence
from typing import override

import torch
from torch import Tensor
from torchdiffeq import odeint

from anypinn.core import (
    ArgsRegistry,
    DataCallback,
    Domain1D,
    GenerationConfig,
    PINNDataModule,
    ValidationRegistry,
)
from anypinn.problems.ode import ODEHyperparameters, ODEProperties

S_KEY = "S"
I_KEY = "I"
BETA_KEY = "beta"
DELTA_KEY = "delta"
N_KEY = "N"
Rt_KEY = "Rt"


def SIR(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    """
    The SIR ODE system.
    $$
    \\begin{align}
    \\frac{dS}{dt} &= -beta * S * I / N \\\\
    \\frac{dI}{dt} &= beta * S * I / N - delta * I \\\\
    \\frac{dR}{dt} &= delta * I \\\\
    \\end{align}
    $$

    Args:
        x: Time variable.
        y: State variables [S, I].
        args: Arguments dictionary (beta, delta, N).

    Returns:
        Derivatives [dS/dt, dI/dt].
    """
    S, I = y
    b, d, N = args[BETA_KEY], args[DELTA_KEY], args[N_KEY]

    dS = -b(x) * S * I / N(x)
    dI = b(x) * S * I / N(x) - d(x) * I
    return torch.stack([dS, dI])


def rSIR(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    """
    The reduced SIR ODE system.
    $$
    \\begin{align}
    \\frac{dS}{dt} &= -delta * R * I \\
    \\frac{dI}{dt} &= delta * (R - 1) * I \\
    \\end{align}
    $$

    dI/dt = delta * (R - 1) * I

    Args:
        x: Time variable.
        y: State variables [I].
        args: Arguments dictionary (delta, Rt).

    Returns:
        Derivatives [dI/dt].
    """
    I = y
    d, Rt = args[DELTA_KEY], args[Rt_KEY]

    dI = d(x) * (Rt(x) - 1) * I
    return dI


class SIRInvDataModule(PINNDataModule):
    """
    DataModule for SIR Inverse problem.
    """

    def __init__(
        self,
        hp: ODEHyperparameters,
        gen_props: ODEProperties | None = None,
        validation: ValidationRegistry | None = None,
        callbacks: Sequence[DataCallback] | None = None,
    ):
        super().__init__(hp, validation, callbacks)
        self.gen_props = gen_props

    @override
    def gen_coll(self, domain: Domain1D) -> Tensor:
        """Generate collocation points."""
        x0 = torch.tensor(domain.x0, dtype=torch.float32)
        x1 = torch.tensor(domain.x1, dtype=torch.float32)

        coll = torch.rand((self.hp.training_data.collocations, 1))
        coll = coll * (torch.log1p(x1) - torch.log1p(x0)) + torch.log1p(x0)
        coll = torch.expm1(coll)
        return coll

    @override
    def gen_data(self, config: GenerationConfig) -> tuple[Tensor, Tensor]:
        """Generate synthetic data."""
        assert self.gen_props is not None, "SIR properties are required to generate data"

        args = self.gen_props.args.copy()
        args.update(config.args_to_train)

        data = odeint(
            lambda x, y: self.gen_props.ode(x, y, args),
            self.gen_props.y0,
            config.x,
        )

        I_true = data[:, 1].clamp_min(0.0)

        I_obs = self._noise(I_true, config.noise_level)

        return config.x.unsqueeze(-1), I_obs.unsqueeze(-1)

    def _noise(self, I_true: Tensor, noise_level: float) -> Tensor:
        if noise_level < 1.0:
            return I_true
        else:
            return torch.poisson(I_true / noise_level) * noise_level
