# Agent file

The purpose of this tile is to describe common mistakes and confusion points that agents might encounter as they work in this project. If you ever encounter something in the project that surprises you, please alert the developer working with you and indicate that this is the case in the AgentMD file to help prevent future agents from having the same issue.

# Environment

- Tools like `just` and `uv` are managed by mise. Run `mise activate bash` (or `mise activate zsh`) before using them.

# Workflow rules

- **Always run `just ci` after any code change.**
- **Always ask before committing a `fix:` or `feat:` commit** — these trigger automated releases (patch and minor respectively).
- **NEVER commit `feat!:` or `BREAKING CHANGE:` without explicit user authorization** — these trigger a major version bump.
- Every new example should include:
  1. Ad-hoc classes in `src/anypinn/catalog/`
  2. Example directory under `examples/`
  3. A canonical scaffold source under `src/anypinn/cli/scaffold/<name>/ode.py` and `config.py`
  4. ODE definitions, constants, keys, and problem factories must match between example and scaffold
  5. Run `just check-scaffold-match` to verify consistency

# Documentation

- **Structure:** The docs use MkDocs Material with 5 navigation tabs: Home, Getting Started, Guides, Catalog, API Reference (plus Community).
- **API reference is hand-curated.** There is no auto-generation plugin. Each public class is rendered via an explicit `:::` directive in `docs/reference/{core,problems,lightning}.md`. When adding a new public class, you **must** add a corresponding `:::` entry to the appropriate reference page.
- **Page icons** are managed in `docs/hooks/page_icons.py`. When adding a new page, add its icon mapping there.
- **Navigation** is defined explicitly in `mkdocs.yml` under `nav:`. New pages must be added to the nav or they won't appear.
- **Content placement rules:**
  - Learning-oriented content (tutorials, walkthroughs) → `docs/getting-started/`
  - Task-oriented content (how to do X) → `docs/guides/`
  - New model templates → `docs/catalog/` (use card grid format in `catalog/index.md`)
  - API documentation → `docs/reference/` (curated `:::` directives, not raw autodoc)
- **Versioning** uses `mike`. Deployment is `mike deploy --push --update-aliases <version> latest`, not `mkdocs gh-deploy`.
- **Build commands:** `just docs` to build, `just docs-serve` to preview locally.
- **Run `just docs` after any docs change** to verify the build succeeds (the build runs in strict mode).

# Scaffold architecture

- Each model has a single canonical `ode.py` and `config.py` in `src/anypinn/cli/scaffold/<name>/`.
- Variant-specific code uses marker comments: `# --- VARIANT: axis/value ---` / `# --- END VARIANT ---`.
- Two independent axes: `source` (synthetic/csv) and `direction` (forward/inverse, PDE models only).
- Suffixed names (e.g., `validation_synthetic`, `create_problem_forward`) are stripped by the generator.
- Same-name variables (e.g., `_training_data`, `_validation`) can be used when suffix stripping isn't needed.
- The canonical file must be valid, lintable Python — all variant code is uncommented.
- The generator (`_generator.py`) extracts the selected variant and removes unused imports.
- Train templates (`train_core.py`, `train_lightning.py`) are shared across all models in `_shared/`.
