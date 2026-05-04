# Catalog

The `anypinn create` CLI ships 16 built-in templates. Each scaffolds a
complete, runnable project with an ODE/PDE definition, hyperparameters, and a
training script.

```bash
anypinn create --list-templates
```

---

## ODE Inverse Problems

<div class="grid cards" markdown>

- :material-virus: **[SIR Epidemic Model](sir.md)**

    ---

    Recovers transmission rate β from infected counts.

    `--template sir`

- :material-virus-outline: **[SEIR Epidemic Model](seir.md)**

    ---

    Recovers transmission rate β with exposed compartment.

    `--template seir`

- :material-sine-wave: **[Damped Oscillator](damped-oscillator.md)**

    ---

    Recovers damping ratio ζ from displacement data.

    `--template damped-oscillator`

- :material-rabbit: **[Lotka-Volterra](lotka-volterra.md)**

    ---

    Recovers predation rate β from predator-prey dynamics.

    `--template lotka-volterra`

- :material-wave: **[Van der Pol Oscillator](van-der-pol.md)**

    ---

    Recovers nonlinearity parameter μ.

    `--template van-der-pol`

- :material-weather-tornado: **[Lorenz System](lorenz.md)**

    ---

    Recovers σ, ρ, β from chaotic trajectory.

    `--template lorenz`

- :material-brain: **[FitzHugh-Nagumo](fitzhugh-nagumo.md)**

    ---

    Recovers timescale ε from neuron spike data.

    `--template fitzhugh-nagumo`

</div>

## PDE Problems

<div class="grid cards" markdown>

- :material-texture-box: **[Gray-Scott 2D](gray-scott-2d.md)**

    ---

    Reaction-diffusion pattern formation.

    `--template gray-scott-2d`

- :material-grid: **[Poisson 2D](poisson-2d.md)**

    ---

    Elliptic PDE forward problem.

    `--template poisson-2d`

- :material-thermometer: **[Heat Equation 1D](heat-1d.md)**

    ---

    Thermal diffusivity recovery.

    `--template heat-1d`

- :material-waves-arrow-right: **[Burgers Equation 1D](burgers-1d.md)**

    ---

    Viscosity recovery with shocks.

    `--template burgers-1d`

- :material-waveform: **[Wave Equation 1D](wave-1d.md)**

    ---

    Wave speed recovery.

    `--template wave-1d`

- :material-blur: **[Inverse Diffusivity](inverse-diffusivity.md)**

    ---

    Function-valued D(x) recovery.

    `--template inverse-diffusivity`

- :material-chart-bell-curve-cumulative: **[Allen-Cahn](allen-cahn.md)**

    ---

    Stiff reaction-diffusion forward problem.

    `--template allen-cahn`

</div>

## Utility Templates

<div class="grid cards" markdown>

- :material-pencil: **[Custom](custom.md)**

    ---

    Stub skeleton for user-defined ODE.

    `--template custom`

- :material-file-outline: **[Blank](blank.md)**

    ---

    Empty project, start from scratch.

    `--template blank`

</div>
