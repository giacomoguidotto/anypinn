"""Catalog of ready-made ODE problem building blocks."""

from pinn.catalog.damped_oscillator import (
    OMEGA_KEY,
    V_KEY,
    X_KEY,
    ZETA_KEY,
    DampedOscillatorDataModule,
)
from pinn.catalog.lotka_volterra import (
    ALPHA_KEY,
    BETA_KEY,
    DELTA_KEY,
    GAMMA_KEY,
    Y_KEY,
    LotkaVolterraDataModule,
)
from pinn.catalog.lotka_volterra import X_KEY as LV_X_KEY
from pinn.catalog.seir import BETA_KEY as SEIR_BETA_KEY
from pinn.catalog.seir import E_KEY, SIGMA_KEY, SEIRDataModule
from pinn.catalog.seir import GAMMA_KEY as SEIR_GAMMA_KEY
from pinn.catalog.seir import I_KEY as SEIR_I_KEY
from pinn.catalog.seir import S_KEY as SEIR_S_KEY
from pinn.catalog.sir import BETA_KEY as SIR_BETA_KEY
from pinn.catalog.sir import DELTA_KEY as SIR_DELTA_KEY
from pinn.catalog.sir import I_KEY as SIR_I_KEY
from pinn.catalog.sir import N_KEY, SIR, Rt_KEY, SIRInvDataModule, rSIR
from pinn.catalog.sir import S_KEY as SIR_S_KEY

__all__ = [
    "ALPHA_KEY",
    "BETA_KEY",
    "DELTA_KEY",
    "E_KEY",
    "GAMMA_KEY",
    "LV_X_KEY",
    "N_KEY",
    "OMEGA_KEY",
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
    "V_KEY",
    "X_KEY",
    "Y_KEY",
    "ZETA_KEY",
    "DampedOscillatorDataModule",
    "LotkaVolterraDataModule",
    "Rt_KEY",
    "SEIRDataModule",
    "SIRInvDataModule",
    "rSIR",
]
