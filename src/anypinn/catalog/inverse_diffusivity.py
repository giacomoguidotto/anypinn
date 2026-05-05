"""Inverse diffusivity problem catalog entry."""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import TYPE_CHECKING, override

import torch
from torch import Tensor

from anypinn.core import GenerationConfig, PINNDataModule, PINNHyperparameters, ValidationRegistry

if TYPE_CHECKING:
    import numpy as np
from anypinn.core.dataset import DataCallback

U_KEY = "u"
D_KEY = "D"


def true_d_fn(x: Tensor) -> Tensor:
    """True diffusivity profile: D(x) = 0.1 + 0.05 sin(2 pi x)."""
    return 0.1 + 0.05 * torch.sin(2 * math.pi * x)


TRUE_D_FN: Callable[[Tensor], Tensor] = true_d_fn


def reference_solution(pts: np.ndarray, n_x: int = 80, n_t: int = 80) -> np.ndarray:
    """Compute clean reference u(x,t) via method-of-lines with the true D(x) profile."""
    import numpy as np
    from scipy.integrate import solve_ivp
    from scipy.interpolate import RegularGridInterpolator

    dx = 1.0 / (n_x - 1)
    x_fd = np.linspace(0, 1, n_x)
    d_vals = 0.1 + 0.05 * np.sin(2 * np.pi * x_fd)
    u0 = np.sin(np.pi * x_fd)
    u0[0] = 0.0
    u0[-1] = 0.0

    def rhs(_t: float, u: np.ndarray) -> np.ndarray:
        du_dt = np.zeros_like(u)
        for i in range(1, n_x - 1):
            d_right = 0.5 * (d_vals[i] + d_vals[i + 1])
            d_left = 0.5 * (d_vals[i - 1] + d_vals[i])
            du_dt[i] = (d_right * (u[i + 1] - u[i]) - d_left * (u[i] - u[i - 1])) / dx**2
        return du_dt

    t_eval = np.linspace(0, 1, n_t)
    sol = solve_ivp(rhs, (0.0, 1.0), u0, t_eval=t_eval, method="Radau", max_step=0.05)
    interp = RegularGridInterpolator(
        (x_fd, sol.t),
        sol.y,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    return interp(pts)


class InverseDiffusivityDataModule(PINNDataModule):
    """DataModule for 1D inverse diffusivity problem.

    gen_data produces ground-truth u(x,t) via scipy method-of-lines
    (central differences with variable D(x), integrated with solve_ivp),
    with optional measurement noise.
    """

    def __init__(
        self,
        hp: PINNHyperparameters,
        n_x: int = 80,
        n_t: int = 80,
        noise_std: float = 0.01,
        grid_size: int = 50,
        validation: ValidationRegistry | None = None,
        callbacks: list[DataCallback] | None = None,
    ):
        self.n_x = n_x
        self.n_t = n_t
        self.noise_std = noise_std
        self.grid_size = grid_size
        super().__init__(hp, validation, callbacks)

    @override
    def gen_data(self, config: GenerationConfig) -> tuple[Tensor, Tensor]:
        """Generate numerical solution on a 2D meshgrid via method of lines."""
        xs = torch.linspace(0, 1, self.grid_size)
        ts = torch.linspace(0, 1, self.grid_size)
        grid_x, grid_t = torch.meshgrid(xs, ts, indexing="ij")
        x_grid = torch.stack([grid_x.reshape(-1), grid_t.reshape(-1)], dim=1)  # (N, 2)

        u_ref = torch.tensor(
            reference_solution(x_grid.numpy(), self.n_x, self.n_t),
            dtype=torch.float32,
        )

        # Add measurement noise
        u_noisy = u_ref + self.noise_std * torch.randn_like(u_ref)

        # Shape: (N, 1, 1) to match codebase convention
        y_data = u_noisy.unsqueeze(-1).unsqueeze(1)

        return x_grid, y_data
