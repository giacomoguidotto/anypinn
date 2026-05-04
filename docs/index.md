---
hide:
  - navigation
  - toc
---

# AnyPINN

**Solve differential equations with Physics-Informed Neural Networks.**

Most PINN libraries make you wire up every loss term, collocation grid, and training loop by hand before you see a single result. AnyPINN gives you a working experiment in one command and then lets you peel back every layer when you're ready.

---

<div class="grid cards" markdown>

- :material-rocket-launch: **One Command Setup**

    ---

    `anypinn create` scaffolds a runnable project from 16 built-in templates.
    No boilerplate, no wiring. Edit physics and press start.

    [:octicons-arrow-right-24: Get started](getting-started/index.md)

- :material-magnify: **Inverse-First**

    ---

    Recover unknown parameters from observations. Promoting a constant to a
    learnable parameter is a one-line change.

    [:octicons-arrow-right-24: How it works](guides/inverse-vs-forward.md)

- :material-puzzle: **Training Agnostic**

    ---

    `Problem` is a plain `nn.Module`. Use PyTorch Lightning, a raw training
    loop, or anything that calls `.backward()`.

    [:octicons-arrow-right-24: Lightning vs Core](guides/lightning-vs-core.md)

</div>

---

## Quick Start

```bash
uvx anypinn create my-project
```

`anypinn create` scaffolds a complete, runnable project with your choice of
template, data source, and training framework.

[:octicons-arrow-right-24: Full walkthrough](getting-started/first-project.md)

---

## Who Is This For?

<div class="grid cards" markdown>

- :material-test-tube: **Experimenter**

    ---

    Run a known problem, tweak parameters, see results. Pick a built-in
    template, change `config.py`, press start.

- :material-flask: **Researcher**

    ---

    Define new physics or custom constraints. Subclass `Constraint` and
    `Problem`, use the provided loss machinery.

- :material-wrench: **Framework Builder**

    ---

    Custom training loops, novel architectures. Use `anypinn.core` directly,
    no Lightning required.

</div>

---

## Results Gallery

<div class="gallery-masonry" markdown>

![Lorenz system](examples/lorenz/results/lorenz.png){ loading=lazy }

![Allen-Cahn equation](examples/allen_cahn/results/allen-cahn.png){ loading=lazy }

![Damped oscillator](examples/damped_oscillator/results/damped-oscillator.png){ loading=lazy }

![Hospitalized SIR inverse problem](examples/hospitalized_sir_inverse/results/hospitalized-sir-inverse.png){ loading=lazy }

![Poisson 2D equation](examples/poisson_2d/results/poisson-2d.png){ loading=lazy }

![FitzHugh-Nagumo model](examples/fitzhugh_nagumo/results/fitzhugh-nagumo.png){ loading=lazy }

![Wave 1D equation](examples/wave_1d/results/wave-1d.png){ loading=lazy }

![SIR inverse problem](examples/sir_inverse/results/sir-inverse.png){ loading=lazy }

![Lotka-Volterra system](examples/lotka_volterra/results/lotka-volterra.png){ loading=lazy }

![Burgers 1D equation](examples/burgers_1d/results/burgers-1d.png){ loading=lazy }

![SEIR inverse problem](examples/seir_inverse/results/seir-inverse.png){ loading=lazy }

![Gray-Scott 2D system](examples/gray_scott_2d/results/gray-scott-2d.png){ loading=lazy }

![Van der Pol oscillator](examples/van_der_pol/results/van-der-pol.png){ loading=lazy }

![Reduced SIR inverse problem](examples/reduced_sir_inverse/results/reduced-sir-inverse.png){ loading=lazy }

![Heat 1D equation](examples/heat_1d/results/heat-1d.png){ loading=lazy }

![Inverse diffusivity problem](examples/inverse_diffusivity/results/inverse-diffusivity.png){ loading=lazy }

</div>

[:octicons-arrow-right-24: Browse all 16 templates](catalog/index.md)
