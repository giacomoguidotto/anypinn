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

## Documentation

The docs live in `docs/` and are built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

### Preview locally

```bash
just docs-serve
```

This starts a hot-reloading server at `http://localhost:8000`. Changes to Markdown files and source docstrings are picked up automatically.

### Where to add content

| Content type | Directory | Example |
| ------------ | --------- | ------- |
| Tutorials, walkthroughs | `docs/getting-started/` | "First Project" |
| Task-oriented how-to guides | `docs/guides/` | "Use CSV Data" |
| New model templates | `docs/catalog/` | "SIR Epidemic Model" |
| API reference | `docs/reference/` | Core, Problems, Lightning |

New pages must also be added to:

1. **`mkdocs.yml`** — the `nav:` section, or the page won't appear in navigation.
2. **`docs/hooks/page_icons.py`** — the `PAGE_ICONS` dict, for sidebar icons.

### API reference

The API reference is **hand-curated**, not auto-generated. Each public class is rendered using an explicit [mkdocstrings](https://mkdocstrings.github.io/) directive:

```markdown
::: anypinn.core.Field
    options:
      show_source: false
      show_signature_annotations: true
```

When you add a new public class, add a corresponding `:::` entry to the appropriate file in `docs/reference/`. Group related classes under section headers.

### Versioning

Docs are versioned with [mike](https://github.com/jimporter/mike). The CI workflow extracts the version from `pyproject.toml` and deploys with `mike deploy`. A version dropdown appears in the header of the deployed site.

### Announcement banner

Use the announcement banner for release highlights or important notices. Add it
to `mkdocs.yml`:

```yaml
extra:
  announcement: "AnyPINN v0.25 is out! <a href='getting-started/'>Get started →</a>"
```

### Status badges

Mark pages as `new`, `deprecated`, or `experimental` by adding a `status` field
to the page's YAML front matter:

```markdown
---
status: new
---

# My New Page
```

This shows a badge next to the page title in the sidebar. Use `new` for recently
added pages and `deprecated` for pages scheduled for removal.

### Verify the build

Always build before pushing docs changes — the build runs in strict mode and will catch broken links and missing references:

```bash
just docs
```

## Conventions

- Commits: [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`)
  - `fix:` triggers a **patch** release (e.g. 0.1.0 → 0.1.1)
  - `feat:` triggers a **minor** release (e.g. 0.1.1 → 0.2.0)
  - `feat!:` or `BREAKING CHANGE:` triggers a **major** release (e.g. 0.2.0 → 1.0.0)
- Branch names: `feat/`, `fix/`, `docs/`, etc.
- Line length: 99
- Absolute imports only — no relative imports
- All config dataclasses: `@dataclass(frozen=True, kw_only=True)`
