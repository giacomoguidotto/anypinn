from __future__ import annotations

from typing import override

import torch
from torch import Tensor

from anypinn.core import GenerationConfig, PINNDataModule, PINNHyperparameters, ValidationRegistry
from anypinn.core.dataset import DataCallback
from anypinn.core.samplers import ResidualScorer

U_KEY = "u"
V_KEY = "v"
DU_KEY = "D_u"
DV_KEY = "D_v"
F_KEY = "F"
K_KEY = "k"

TRUE_DU = 5e-3
TRUE_DV = 2.5e-3
TRUE_F = 0.04
TRUE_K = 0.06

T_TOTAL = 200


class GrayScott2DDataModule(PINNDataModule):
    """DataModule for 2D Gray-Scott reaction-diffusion inverse problem.

    gen_data produces ground-truth u(x,y,t) and v(x,y,t) via scipy
    method-of-lines (finite-difference spatial discretization + ODE
    integration), with optional measurement noise.
    """

    def __init__(
        self,
        hp: PINNHyperparameters,
        true_du: float = TRUE_DU,
        true_dv: float = TRUE_DV,
        true_f: float = TRUE_F,
        true_k: float = TRUE_K,
        noise_std: float = 0.01,
        sim_size: int = 64,
        residual_scorer: ResidualScorer | None = None,
        validation: ValidationRegistry | None = None,
        callbacks: list[DataCallback] | None = None,
    ):
        self.true_du = true_du
        self.true_dv = true_dv
        self.true_f = true_f
        self.true_k = true_k
        self.noise_std = noise_std
        self.sim_size = sim_size
        super().__init__(hp, validation, callbacks, residual_scorer=residual_scorer)

    @override
    def gen_data(self, config: GenerationConfig) -> tuple[Tensor, Tensor]:
        """Generate numerical solution on a 3D meshgrid via method of lines."""
        import numpy as np
        from scipy.integrate import solve_ivp
        from scipy.interpolate import RegularGridInterpolator

        n = self.sim_size
        dx = 1.0 / (n - 1)
        x_fd = np.linspace(0, 1, n)

        # ICs: u=1, v=0 everywhere; center square u=0.5, v=0.25
        u0 = np.ones((n, n))
        v0 = np.zeros((n, n))
        lo = int(0.4 * n)
        hi = int(0.6 * n)
        u0[lo:hi, lo:hi] = 0.5
        v0[lo:hi, lo:hi] = 0.25

        y0 = np.concatenate([u0.ravel(), v0.ravel()])

        du_val = self.true_du
        dv_val = self.true_dv
        f_val = self.true_f
        k_val = self.true_k

        def rhs(_t: float, y: np.ndarray) -> np.ndarray:
            u = y[: n * n].reshape(n, n)
            v = y[n * n :].reshape(n, n)

            # 5-point FD Laplacian with Neumann (zero-flux) BCs via padding
            u_pad = np.pad(u, 1, mode="edge")
            v_pad = np.pad(v, 1, mode="edge")

            lap_u = (
                u_pad[2:, 1:-1]
                + u_pad[:-2, 1:-1]
                + u_pad[1:-1, 2:]
                + u_pad[1:-1, :-2]
                - 4 * u_pad[1:-1, 1:-1]
            ) / dx**2
            lap_v = (
                v_pad[2:, 1:-1]
                + v_pad[:-2, 1:-1]
                + v_pad[1:-1, 2:]
                + v_pad[1:-1, :-2]
                - 4 * v_pad[1:-1, 1:-1]
            ) / dx**2

            uv2 = u * v**2
            du_dt = du_val * lap_u - uv2 + f_val * (1 - u)
            dv_dt = dv_val * lap_v + uv2 - (f_val + k_val) * v

            return np.concatenate([du_dt.ravel(), dv_dt.ravel()])

        t_span = (0.0, T_TOTAL)
        n_t_sim = 50
        t_eval = np.linspace(0, T_TOTAL, n_t_sim)

        sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, method="Radau", max_step=5.0)

        # sol.y shape: (2*n*n, n_t_sim)
        u_sol = sol.y[: n * n, :].reshape(n, n, -1)  # (n, n, n_t)
        v_sol = sol.y[n * n :, :].reshape(n, n, -1)

        # Build interpolators for u and v
        t_norm_sim = sol.t / T_TOTAL  # normalize to [0, 1]
        interp_u = RegularGridInterpolator(
            (x_fd, x_fd, t_norm_sim),
            u_sol,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        interp_v = RegularGridInterpolator(
            (x_fd, x_fd, t_norm_sim),
            v_sol,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        # Output measurement grid in [0,1]^3
        n_xy = 20
        n_t = 10
        xs = torch.linspace(0, 1, n_xy)
        ys = torch.linspace(0, 1, n_xy)
        ts = torch.linspace(0, 1, n_t)
        gx, gy, gt = torch.meshgrid(xs, ys, ts, indexing="ij")
        x_grid = torch.stack([gx.reshape(-1), gy.reshape(-1), gt.reshape(-1)], dim=1)

        pts = x_grid.numpy()
        u_ref = torch.tensor(interp_u(pts), dtype=torch.float32)
        v_ref = torch.tensor(interp_v(pts), dtype=torch.float32)

        # Add measurement noise
        u_noisy = u_ref + self.noise_std * torch.randn_like(u_ref)
        v_noisy = v_ref + self.noise_std * torch.randn_like(v_ref)

        # Shape: (N, 2, 1)
        y_data = torch.stack([u_noisy, v_noisy], dim=1).unsqueeze(-1)

        return x_grid, y_data
