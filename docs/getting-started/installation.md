# Installation

AnyPINN requires **Python 3.10 or later**.

---

## Install the CLI

The `anypinn` CLI is the main entry point. Install it globally so it's available
from any directory:

=== "uv"

    ```bash
    uv tool install anypinn
    ```

=== "pip"

    ```bash
    pip install anypinn
    ```

Verify the installation:

```bash
anypinn --version
```

!!! tip "Use `uvx` for one-off runs"

    If you don't want to install globally, you can run the CLI directly with
    [uvx](https://docs.astral.sh/uv/guides/tools/):

    ```bash
    uvx anypinn create my-project
    ```

---

## Dependencies

Each project scaffolded by `anypinn create` includes its own `pyproject.toml`
with pinned dependencies. Running `uv sync` inside the project directory
installs everything you need:

- **PyTorch** — tensor computation and automatic differentiation
- **NumPy / SciPy** — numerical utilities
- **torchdiffeq** — ODE integration for synthetic data generation
- **PyTorch Lightning** — optional, included when `--lightning` is selected
- **Pandas** — CSV data loading
- **Matplotlib / Seaborn** — result visualization

You do not need to install these manually — the generated project handles it.

---

## GPU Support

AnyPINN uses PyTorch under the hood. If you have a CUDA-capable GPU, install
the appropriate PyTorch build for your system. See the
[PyTorch installation guide](https://pytorch.org/get-started/locally/) for
platform-specific instructions.

When using Lightning, GPU training is automatic — the `Trainer` detects
available hardware and uses it.
