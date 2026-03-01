from __future__ import annotations

import math
from typing import override

import torch
from torch import Tensor

from anypinn.core import GenerationConfig, PINNDataModule, PINNHyperparameters, ValidationRegistry
from anypinn.core.dataset import DataCallback

U_KEY = "u"
C_KEY = "c"

TRUE_C = 1.0


class Wave1DDataModule(PINNDataModule):
    """DataModule for 1D wave equation inverse problem.

    gen_data produces sparse interior measurements from the analytic
    solution u(x,t) = sin(pi x) cos(c pi t), with optional noise.
    These measurements are used by DataConstraint during training
    to recover c.
    """

    def __init__(
        self,
        hp: PINNHyperparameters,
        true_c: float = TRUE_C,
        n_measurements: int = 200,
        noise_std: float = 0.01,
        grid_size: int = 50,
        validation: ValidationRegistry | None = None,
        callbacks: list[DataCallback] | None = None,
    ):
        self.true_c = true_c
        self.n_measurements = n_measurements
        self.noise_std = noise_std
        self.grid_size = grid_size
        super().__init__(hp, validation, callbacks)

    @override
    def gen_data(self, config: GenerationConfig) -> tuple[Tensor, Tensor]:
        """Generate analytic solution on a 2D meshgrid for training + prediction."""
        xs = torch.linspace(0, 1, self.grid_size)
        ts = torch.linspace(0, 1, self.grid_size)
        grid_x, grid_t = torch.meshgrid(xs, ts, indexing="ij")

        x_grid = torch.stack([grid_x.reshape(-1), grid_t.reshape(-1)], dim=1)  # (N, 2)

        u_analytic = torch.sin(math.pi * x_grid[:, 0]) * torch.cos(
            self.true_c * math.pi * x_grid[:, 1]
        )

        # Add measurement noise
        u_noisy = u_analytic + self.noise_std * torch.randn_like(u_analytic)

        # Shape: (N, 1, 1) to match codebase convention
        y_data = u_noisy.unsqueeze(-1).unsqueeze(1)

        return x_grid, y_data
