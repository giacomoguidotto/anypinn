from __future__ import annotations

import math
from typing import override

import torch
from torch import Tensor

from anypinn.core import GenerationConfig, PINNDataModule, PINNHyperparameters, ValidationRegistry
from anypinn.core.dataset import DataCallback

U_KEY = "u"


class Poisson2DDataModule(PINNDataModule):
    """DataModule for 2D Poisson equation on [0,1]^2.

    Generates a meshgrid with the analytic solution u(x,y) = sin(pi*x)*sin(pi*y).
    The data is used for prediction/validation only -- training uses
    PDEResidualConstraint + DirichletBCConstraints (no DataConstraint).
    """

    def __init__(
        self,
        hp: PINNHyperparameters,
        grid_size: int = 30,
        validation: ValidationRegistry | None = None,
        callbacks: list[DataCallback] | None = None,
    ):
        self.grid_size = grid_size
        super().__init__(hp, validation, callbacks)

    @override
    def gen_data(self, config: GenerationConfig) -> tuple[Tensor, Tensor]:
        """Generate analytic solution on a 2D meshgrid for prediction."""
        xs = torch.linspace(0, 1, self.grid_size)
        ys = torch.linspace(0, 1, self.grid_size)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")

        x_grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)  # (N, 2)

        u_analytic = torch.sin(math.pi * x_grid[:, 0]) * torch.sin(math.pi * x_grid[:, 1])

        # Shape: (N, 1, 1) to match codebase convention
        y_data = u_analytic.unsqueeze(-1).unsqueeze(1)

        return x_grid, y_data
