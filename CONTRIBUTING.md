# Contributing

Welcome! Contributions are warmly appreciated — bug reports, new problem templates, constraint types, documentation improvements, and more.

By participating in this project you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## 🛠️ Setup

> **Prerequisites:** Python 3.11+ with
> [uv](https://github.com/astral-sh/uv) and
> [just](https://github.com/casey/just).

```bash
git clone https://github.com/your-org/anypinn
cd anypinn
uv sync
```

> **Optional: `mise`**
> If you use [mise](https://mise.jdx.dev), run `mise install` first to provision the
> pinned `python`, `uv`, and `just` versions declared in `.mise.toml`.
>
> The project still uses the standard `uv` virtual environment layout, so `ty` works
> from the shared `pyproject.toml` configuration without any extra local overrides.

All common tasks are driven by `just`:

```bash
just test           # Run tests with coverage
just lint           # Check code style
just fmt            # Format code (isort + ruff)
just lint-fix       # Auto-fix linting issues
just check          # Type checking (ty)
just docs-serve     # Serve docs locally
just ci             # lint + check + test (full CI suite)
```

## 🚀 Running Examples

The `examples/` directory ships ready-to-run problems you can use to exercise the
library end-to-end while developing.

### Lightning examples (most problems)

Each example lives in its own directory with a `train.py` entry-point:

```bash
cd examples/lotka_volterra        # or any other example directory
uv run python train.py            # train from scratch
uv run python train.py --predict  # load the saved checkpoint and predict only
```

Training logs are written to `examples/_logs/` (TensorBoard + CSV) and model
artefacts are saved under `examples/<name>/results/`.

### Core-only example

`exponential_decay` is a minimal, single-file example that uses only
`anypinn.core` (no Lightning):

```bash
uv run python examples/exponential_decay/exponential_decay.py
```

### Scaffolding a new example

The CLI can generate a fresh project from a built-in template:

```bash
uv run anypinn create my_project --template lotka-volterra --data synthetic
```

Run `uv run anypinn create --list-templates` to see all available templates
(`sir`, `poisson-2d`, `heat-1d`, `gray-scott-2d`, `blank`, etc.).

## 🔁 Workflow

1. Fork the repository and create a branch for your change:

   ```bash
   git checkout -b feat/my-feature
   ```

2. Make your changes, then verify everything passes:

   ```bash
   just ci
   ```

3. Commit following [Conventional Commits](https://www.conventionalcommits.org/):

   ```bash
   git commit -m "feat: add Brusselator ODE template"
   ```

   | Prefix                        | Effect                |
   | ----------------------------- | --------------------- |
   | `fix:`                        | Patch release (0.0.X) |
   | `feat:`                       | Minor release (0.X.0) |
   | `feat!:` / `BREAKING CHANGE:` | Major release (X.0.0) |

4. Push and open a pull request.

## ✍️ Code Style

- Line length: 99
- Ruff linter with rules: F, E, I, N, UP, RUF, B, C4, ISC, PIE, PT, PTH, SIM, TID
- Absolute imports only — no relative imports
- All config dataclasses: `@dataclass(frozen=True, kw_only=True)`

## 🏗️ Architecture Guidelines

- Keep the layer separation: `anypinn.core` stays pure PyTorch, Lightning stays optional.
- `anypinn.core` must not import from `anypinn.lightning`, `anypinn.problems`, or `anypinn.catalog`.
- If you change the architecture or data flow, update both `CLAUDE.md` and `README.md`.
