from __future__ import annotations

import math
from typing import override

import torch
from torch import Tensor

from anypinn.core import GenerationConfig, PINNDataModule, PINNHyperparameters, ValidationRegistry
from anypinn.core.dataset import DataCallback
from anypinn.core.samplers import ResidualScorer

U_KEY = "u"
NU_KEY = "nu"

TRUE_NU = 0.01 / math.pi


class Burgers1DDataModule(PINNDataModule):
    """DataModule for 1D Burgers equation inverse problem.

    gen_data produces ground-truth u(x,t) via scipy method-of-lines
    (finite-difference spatial discretization + ODE integration),
    with optional measurement noise.
    """

    def __init__(
        self,
        hp: PINNHyperparameters,
        true_nu: float = TRUE_NU,
        noise_std: float = 0.01,
        grid_size: int = 50,
        residual_scorer: ResidualScorer | None = None,
        validation: ValidationRegistry | None = None,
        callbacks: list[DataCallback] | None = None,
    ):
        self.true_nu = true_nu
        self.noise_std = noise_std
        self.grid_size = grid_size
        super().__init__(hp, validation, callbacks, residual_scorer=residual_scorer)

    @override
    def gen_data(self, config: GenerationConfig) -> tuple[Tensor, Tensor]:
        """Generate numerical solution on a 2D meshgrid via method of lines."""
        import numpy as np
        from scipy.integrate import solve_ivp
        from scipy.interpolate import RegularGridInterpolator

        n_x = 256  # interior spatial points for FD discretization
        x_fd = torch.linspace(-1, 1, n_x + 2).numpy()  # includes boundaries
        dx = x_fd[1] - x_fd[0]

        # IC: u(x,0) = -sin(pi*x)
        u0 = -torch.sin(math.pi * torch.tensor(x_fd[1:-1])).numpy()

        t_span = (0.0, 1.0)
        t_eval = torch.linspace(0, 1, self.grid_size).numpy()

        nu = self.true_nu

        def rhs(_t, u):
            """RHS: du/dt = nu*d2u/dx2 - u*du/dx with homogeneous Dirichlet BCs."""
            # Pad with boundary values (u=0 at x=-1 and x=1)
            u_pad = torch.zeros(len(u) + 2).numpy()
            u_pad[1:-1] = u

            # Central differences for d2u/dx2
            d2u = (u_pad[2:] - 2 * u_pad[1:-1] + u_pad[:-2]) / dx**2

            # Central differences for du/dx
            du = (u_pad[2:] - u_pad[:-2]) / (2 * dx)

            return nu * d2u - u * du

        sol = solve_ivp(rhs, t_span, u0, t_eval=t_eval, method="RK45", max_step=0.001)

        # sol.y has shape (n_x, len(t_eval))
        # Interpolate onto output meshgrid
        x_interior = x_fd[1:-1]
        t_sol = sol.t

        interp = RegularGridInterpolator(
            (x_interior, t_sol), sol.y, method="linear", bounds_error=False, fill_value=None
        )

        # Output meshgrid
        xs = torch.linspace(-1, 1, self.grid_size)
        ts = torch.linspace(0, 1, self.grid_size)
        grid_x, grid_t = torch.meshgrid(xs, ts, indexing="ij")

        x_grid = torch.stack([grid_x.reshape(-1), grid_t.reshape(-1)], dim=1)  # (N, 2)

        pts = np.stack([x_grid[:, 0].numpy(), x_grid[:, 1].numpy()], axis=1)
        u_ref = torch.tensor(interp(pts), dtype=torch.float32)

        # Add measurement noise
        u_noisy = u_ref + self.noise_std * torch.randn_like(u_ref)

        # Shape: (N, 1, 1) to match codebase convention
        y_data = u_noisy.unsqueeze(-1).unsqueeze(1)

        return x_grid, y_data
