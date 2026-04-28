# Catalog

The `anypinn create` CLI ships 16 built-in catalog entries. Each scaffolds a complete, runnable
project with an ODE/PDE definition, hyperparameters, and a training script.

Run `anypinn create --list-templates` to see the full list from the command line.

---

## ODE Inverse Problems

| Template | CLI flag | Recovers |
| -------- | -------- | -------- |
| [SIR Epidemic Model](sir.md) | `--template sir` | Transmission rate β |
| [SEIR Epidemic Model](seir.md) | `--template seir` | Transmission rate β |
| [Damped Oscillator](damped-oscillator.md) | `--template damped-oscillator` | Damping ratio ζ |
| [Lotka-Volterra](lotka-volterra.md) | `--template lotka-volterra` | Predation rate β |
| [Van der Pol Oscillator](van-der-pol.md) | `--template van-der-pol` | Nonlinearity μ |
| [Lorenz System](lorenz.md) | `--template lorenz` | σ, ρ, β |
| [FitzHugh-Nagumo](fitzhugh-nagumo.md) | `--template fitzhugh-nagumo` | Timescale ε |

## PDE Problems

| Template | CLI flag | Description |
| -------- | -------- | ----------- |
| [Gray-Scott 2D](gray-scott-2d.md) | `--template gray-scott-2d` | Reaction-diffusion pattern formation |
| [Poisson 2D](poisson-2d.md) | `--template poisson-2d` | Elliptic PDE forward problem |
| [Heat Equation 1D](heat-1d.md) | `--template heat-1d` | Thermal diffusivity recovery |
| [Burgers Equation 1D](burgers-1d.md) | `--template burgers-1d` | Viscosity recovery with shocks |
| [Wave Equation 1D](wave-1d.md) | `--template wave-1d` | Wave speed recovery |
| [Inverse Diffusivity](inverse-diffusivity.md) | `--template inverse-diffusivity` | Function-valued D(x) recovery |
| [Allen-Cahn](allen-cahn.md) | `--template allen-cahn` | Stiff reaction-diffusion forward |

## Utility Templates

| Template | CLI flag | Description |
| -------- | -------- | ----------- |
| [Custom ODE](custom.md) | `--template custom` | Stub skeleton for user-defined ODE |
| [Blank](blank.md) | `--template blank` | Empty project, start from scratch |
