from __future__ import annotations

import math
from typing import override

import torch
from torch import Tensor

from anypinn.core import GenerationConfig, PINNDataModule, PINNHyperparameters, ValidationRegistry
from anypinn.core.dataset import DataCallback
from anypinn.core.samplers import ResidualScorer

U_KEY = "u"

TRUE_EPSILON = 0.01


class AllenCahnDataModule(PINNDataModule):
    """DataModule for 1D Allen-Cahn equation.

    gen_data produces ground-truth u(x,t) via scipy method-of-lines
    (central differences for d2u/dx2 with periodic ghost cells + ODE integration).
    The data is used for prediction/validation only — training uses
    PDEResidualConstraint + PeriodicBCConstraint + IC (no DataConstraint).
    """

    def __init__(
        self,
        hp: PINNHyperparameters,
        true_epsilon: float = TRUE_EPSILON,
        n_x: int = 256,
        n_t: int = 200,
        grid_size: int = 50,
        residual_scorer: ResidualScorer | None = None,
        validation: ValidationRegistry | None = None,
        callbacks: list[DataCallback] | None = None,
    ):
        self.true_epsilon = true_epsilon
        self.n_x = n_x
        self.n_t = n_t
        self.grid_size = grid_size
        super().__init__(hp, validation, callbacks, residual_scorer=residual_scorer)

    @override
    def gen_data(self, config: GenerationConfig) -> tuple[Tensor, Tensor]:
        """Generate numerical solution on a 2D meshgrid via method of lines."""
        import numpy as np
        from scipy.integrate import solve_ivp
        from scipy.interpolate import RegularGridInterpolator

        n_x = self.n_x
        eps = self.true_epsilon

        # Periodic domain [-1, 1): n_x interior points, dx = 2/n_x
        x_fd = np.linspace(-1, 1, n_x, endpoint=False)
        dx = x_fd[1] - x_fd[0]

        # IC: u(x,0) = -tanh(x / (sqrt(2*eps)))
        scale = math.sqrt(2 * eps)
        u0 = -np.tanh(x_fd / scale)

        t_span = (0.0, 1.0)
        t_eval = np.linspace(0, 1, self.n_t)

        def rhs(_t: float, u: np.ndarray) -> np.ndarray:
            """RHS: du/dt = eps*d2u/dx2 + u - u^3 with periodic BCs."""
            # Periodic padding
            u_pad = np.empty(n_x + 2)
            u_pad[1:-1] = u
            u_pad[0] = u[-1]  # left ghost = rightmost interior
            u_pad[-1] = u[0]  # right ghost = leftmost interior

            # Central differences for d2u/dx2
            d2u = (u_pad[2:] - 2 * u_pad[1:-1] + u_pad[:-2]) / dx**2

            return eps * d2u + u - u**3

        sol = solve_ivp(rhs, t_span, u0, t_eval=t_eval, method="RK45", max_step=0.001)

        # sol.y has shape (n_x, n_t)
        x_sol = x_fd
        t_sol = sol.t

        interp = RegularGridInterpolator(
            (x_sol, t_sol), sol.y, method="linear", bounds_error=False, fill_value=None
        )

        # Output meshgrid
        xs = torch.linspace(-1, 1, self.grid_size)
        ts = torch.linspace(0, 1, self.grid_size)
        grid_x, grid_t = torch.meshgrid(xs, ts, indexing="ij")

        x_grid = torch.stack([grid_x.reshape(-1), grid_t.reshape(-1)], dim=1)  # (N, 2)

        pts = np.stack([x_grid[:, 0].numpy(), x_grid[:, 1].numpy()], axis=1)
        u_ref = torch.tensor(interp(pts), dtype=torch.float32)

        # Shape: (N, 1, 1) to match codebase convention
        y_data = u_ref.unsqueeze(-1).unsqueeze(1)

        return x_grid, y_data
