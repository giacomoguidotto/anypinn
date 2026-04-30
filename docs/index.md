---
hide:
  - navigation
  - toc
---

# AnyPINN

**Solve differential equations with Physics-Informed Neural Networks.**

Modular. Training-agnostic. Inverse-problem-first.

---

<div class="grid cards" markdown>

- :material-rocket-launch: **One Command Setup**

    ---

    `anypinn create` scaffolds a runnable project from 16 built-in templates.
    No boilerplate, no wiring — edit physics and press start.

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

## Quick Install

=== "uv"

    ```bash
    uv tool install anypinn
    ```

=== "pip"

    ```bash
    pip install anypinn
    ```

## Quick Start

```bash
anypinn create my-project --template sir --data synthetic --lightning
cd my-project && uv sync && uv run train.py
```

That's it — a working SIR epidemic model recovering transmission rate beta from
synthetic observations, training with PyTorch Lightning, logging to TensorBoard.

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

<div class="grid" markdown>

![Allen-Cahn equation](examples/allen_cahn/results/allen-cahn.png){ loading=lazy }

![Lorenz system](examples/lorenz/results/lorenz.png){ loading=lazy }

![SIR inverse problem](examples/sir_inverse/results/sir-inverse.png){ loading=lazy }

</div>

[:octicons-arrow-right-24: Browse all 16 templates](catalog/index.md)
