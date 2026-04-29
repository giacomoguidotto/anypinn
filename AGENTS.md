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

# Scaffold architecture

- Each model has a single canonical `ode.py` and `config.py` in `src/anypinn/cli/scaffold/<name>/`.
- Variant-specific code uses marker comments: `# --- VARIANT: axis/value ---` / `# --- END VARIANT ---`.
- Two independent axes: `source` (synthetic/csv) and `direction` (forward/inverse, PDE models only).
- Suffixed names (e.g., `validation_synthetic`, `create_problem_forward`) are stripped by the generator.
- Same-name variables (e.g., `_training_data`, `_validation`) can be used when suffix stripping isn't needed.
- The canonical file must be valid, lintable Python — all variant code is uncommented.
- The generator (`_generator.py`) extracts the selected variant and removes unused imports.
- Train templates (`train_core.py`, `train_lightning.py`) are shared across all models in `_shared/`.
