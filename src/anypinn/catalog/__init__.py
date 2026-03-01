"""Catalog of ready-made ODE/PDE problem building blocks."""

from anypinn.catalog.allen_cahn import U_KEY as AC_U_KEY
from anypinn.catalog.allen_cahn import AllenCahnDataModule
from anypinn.catalog.burgers_1d import NU_KEY as BURGERS_NU_KEY
from anypinn.catalog.burgers_1d import U_KEY as BURGERS_U_KEY
from anypinn.catalog.burgers_1d import Burgers1DDataModule
from anypinn.catalog.damped_oscillator import (
    OMEGA_KEY,
    V_KEY,
    X_KEY,
    ZETA_KEY,
    DampedOscillatorDataModule,
)
from anypinn.catalog.fitzhugh_nagumo import A_KEY as FHN_A_KEY
from anypinn.catalog.fitzhugh_nagumo import EPSILON_KEY as FHN_EPSILON_KEY
from anypinn.catalog.fitzhugh_nagumo import V_KEY as FHN_V_KEY
from anypinn.catalog.fitzhugh_nagumo import W_KEY as FHN_W_KEY
from anypinn.catalog.fitzhugh_nagumo import FitzHughNagumoDataModule
from anypinn.catalog.gray_scott_2d import DU_KEY as GS_DU_KEY
from anypinn.catalog.gray_scott_2d import DV_KEY as GS_DV_KEY
from anypinn.catalog.gray_scott_2d import F_KEY as GS_F_KEY
from anypinn.catalog.gray_scott_2d import K_KEY as GS_K_KEY
from anypinn.catalog.gray_scott_2d import U_KEY as GS_U_KEY
from anypinn.catalog.gray_scott_2d import V_KEY as GS_V_KEY
from anypinn.catalog.gray_scott_2d import GrayScott2DDataModule
from anypinn.catalog.heat_1d import ALPHA_KEY as HEAT_ALPHA_KEY
from anypinn.catalog.heat_1d import U_KEY as HEAT_U_KEY
from anypinn.catalog.heat_1d import Heat1DDataModule
from anypinn.catalog.inverse_diffusivity import D_KEY as DIFFUSIVITY_D_KEY
from anypinn.catalog.inverse_diffusivity import U_KEY as DIFFUSIVITY_U_KEY
from anypinn.catalog.inverse_diffusivity import InverseDiffusivityDataModule
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
from anypinn.catalog.wave_1d import C_KEY as WAVE_C_KEY
from anypinn.catalog.wave_1d import U_KEY as WAVE_U_KEY
from anypinn.catalog.wave_1d import Wave1DDataModule

__all__ = [
    "AC_U_KEY",
    "ALPHA_KEY",
    "BETA_KEY",
    "BURGERS_NU_KEY",
    "BURGERS_U_KEY",
    "DELTA_KEY",
    "DIFFUSIVITY_D_KEY",
    "DIFFUSIVITY_U_KEY",
    "E_KEY",
    "FHN_A_KEY",
    "FHN_EPSILON_KEY",
    "FHN_V_KEY",
    "FHN_W_KEY",
    "GAMMA_KEY",
    "GS_DU_KEY",
    "GS_DV_KEY",
    "GS_F_KEY",
    "GS_K_KEY",
    "GS_U_KEY",
    "GS_V_KEY",
    "HEAT_ALPHA_KEY",
    "HEAT_U_KEY",
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
    "WAVE_C_KEY",
    "WAVE_U_KEY",
    "X_KEY",
    "Y_KEY",
    "ZETA_KEY",
    "Z_KEY",
    "AllenCahnDataModule",
    "Burgers1DDataModule",
    "DampedOscillatorDataModule",
    "FitzHughNagumoDataModule",
    "GrayScott2DDataModule",
    "Heat1DDataModule",
    "InverseDiffusivityDataModule",
    "LorenzDataModule",
    "LotkaVolterraDataModule",
    "Poisson2DDataModule",
    "Rt_KEY",
    "SEIRDataModule",
    "SIRInvDataModule",
    "VanDerPolDataModule",
    "Wave1DDataModule",
    "rSIR",
]
