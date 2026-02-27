"""Catalog of ready-made ODE/PDE problem building blocks."""

from anypinn.catalog.damped_oscillator import (
    OMEGA_KEY,
    V_KEY,
    X_KEY,
    ZETA_KEY,
    DampedOscillatorDataModule,
)
from anypinn.catalog.lorenz import BETA_KEY as LORENZ_BETA_KEY
from anypinn.catalog.lorenz import RHO_KEY as LORENZ_RHO_KEY
from anypinn.catalog.lorenz import SIGMA_KEY as LORENZ_SIGMA_KEY
from anypinn.catalog.lorenz import X_KEY as LORENZ_X_KEY
from anypinn.catalog.lorenz import Y_KEY as LORENZ_Y_KEY
from anypinn.catalog.lorenz import Z_KEY, LorenzDataModule
from anypinn.catalog.lotka_volterra import (
    ALPHA_KEY,
    BETA_KEY,
    DELTA_KEY,
    GAMMA_KEY,
    Y_KEY,
    LotkaVolterraDataModule,
)
from anypinn.catalog.lotka_volterra import X_KEY as LV_X_KEY
from anypinn.catalog.poisson_2d import U_KEY as POISSON_U_KEY
from anypinn.catalog.poisson_2d import Poisson2DDataModule
from anypinn.catalog.seir import BETA_KEY as SEIR_BETA_KEY
from anypinn.catalog.seir import E_KEY, SIGMA_KEY, SEIRDataModule
from anypinn.catalog.seir import GAMMA_KEY as SEIR_GAMMA_KEY
from anypinn.catalog.seir import I_KEY as SEIR_I_KEY
from anypinn.catalog.seir import S_KEY as SEIR_S_KEY
from anypinn.catalog.sir import BETA_KEY as SIR_BETA_KEY
from anypinn.catalog.sir import DELTA_KEY as SIR_DELTA_KEY
from anypinn.catalog.sir import I_KEY as SIR_I_KEY
from anypinn.catalog.sir import N_KEY, SIR, Rt_KEY, SIRInvDataModule, rSIR
from anypinn.catalog.sir import S_KEY as SIR_S_KEY
from anypinn.catalog.van_der_pol import MU_KEY, U_KEY, VanDerPolDataModule

__all__ = [
    "ALPHA_KEY",
    "BETA_KEY",
    "DELTA_KEY",
    "E_KEY",
    "GAMMA_KEY",
    "LORENZ_BETA_KEY",
    "LORENZ_RHO_KEY",
    "LORENZ_SIGMA_KEY",
    "LORENZ_X_KEY",
    "LORENZ_Y_KEY",
    "LV_X_KEY",
    "MU_KEY",
    "N_KEY",
    "OMEGA_KEY",
    "POISSON_U_KEY",
    "SEIR_BETA_KEY",
    "SEIR_GAMMA_KEY",
    "SEIR_I_KEY",
    "SEIR_S_KEY",
    "SIGMA_KEY",
    "SIR",
    "SIR_BETA_KEY",
    "SIR_DELTA_KEY",
    "SIR_I_KEY",
    "SIR_S_KEY",
    "U_KEY",
    "V_KEY",
    "X_KEY",
    "Y_KEY",
    "ZETA_KEY",
    "Z_KEY",
    "DampedOscillatorDataModule",
    "LorenzDataModule",
    "LotkaVolterraDataModule",
    "Poisson2DDataModule",
    "Rt_KEY",
    "SEIRDataModule",
    "SIRInvDataModule",
    "VanDerPolDataModule",
    "rSIR",
]
