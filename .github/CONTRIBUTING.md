# Contributing

Welcome! Contributions are warmly appreciated — bug reports, new problem templates, constraint types, documentation improvements, and more.

## 🛠️ Setup

> **Prerequisites:** Python 3.11+, [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/your-org/anypinn
cd anypinn
uv sync
```

> **devenv users:** [devenv](https://devenv.sh) redirects `uv sync` installs to `.devenv/state/venv` instead of the standard `.venv`, so ty cannot auto-discover it. Create a gitignored `ty.toml` at the project root with:
> ```toml
> [environment]
> python-version = "3.13"
> python = "./.devenv/state/venv"
> root = ["./src"]
> ```
> (`ty.toml` takes full precedence over `pyproject.toml`, so all three settings are required.)

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

   | Prefix | Effect |
   |--------|--------|
   | `fix:` | Patch release (0.0.X) |
   | `feat:` | Minor release (0.X.0) |
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
