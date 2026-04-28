# Contributing

Thanks for wanting to contribute to AnyPINN! Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before getting started.

## Setup

1. Fork and clone the repo, then install dependencies:

    ```bash
    uv sync
    ```

    > **Optional:** If you use [mise](https://mise.jdx.dev), run `mise install` first to
    > provision the pinned `python`, `uv`, and `just` versions from `.mise.toml`.

2. Before pushing, run the full CI check locally:

    ```bash
    just ci
    ```

    This runs lint, typecheck, and tests — the same pipeline as CI.

## Tooling

- **Package manager**: [uv](https://github.com/astral-sh/uv)
- **Task runner**: [just](https://github.com/casey/just)
- **Linting & formatting**: [Ruff](https://github.com/astral-sh/ruff)
- **Type checking**: [ty](https://docs.astral.sh/ty/)
- **Tests**: [pytest](https://docs.pytest.org/)

## Good to Know

- The `examples/` directory has ready-to-run problems you can use to exercise the library end-to-end while developing. Each example has a `train.py` entry-point (`uv run python train.py`).
- `exponential_decay` is a minimal, core-only example (no Lightning).
- `uv run anypinn create --list-templates` lists all scaffold templates.
- Keep the layer separation: `anypinn.core` stays pure PyTorch, Lightning stays optional. `anypinn.core` must not import from `anypinn.lightning`, `anypinn.problems`, or `anypinn.catalog`.
- If you change the architecture or data flow, update both `CLAUDE.md` and `README.md`.

## Conventions

- Commits: [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`)
  - `fix:` triggers a **patch** release (e.g. 0.1.0 → 0.1.1)
  - `feat:` triggers a **minor** release (e.g. 0.1.1 → 0.2.0)
  - `feat!:` or `BREAKING CHANGE:` triggers a **major** release (e.g. 0.2.0 → 1.0.0)
- Branch names: `feat/`, `fix/`, `docs/`, etc.
- Line length: 99
- Absolute imports only — no relative imports
- All config dataclasses: `@dataclass(frozen=True, kw_only=True)`
