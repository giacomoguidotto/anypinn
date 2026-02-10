# CHANGELOG


## v0.2.0 (2026-02-10)

### Build System

- Derive version from git tags via uv-dynamic-versioning
  ([`2105d38`](https://github.com/giacomoguidotto/anypinn/commit/2105d38602f2674bd515b9ce4f7bdbd07ac762fc))

Switch build backend from uv_build to hatchling + uv-dynamic-versioning so semantic-release only
  creates a git tag without a version-bump commit. Version is now read from importlib.metadata at
  runtime.

### Documentation

- Add contributing guidelines
  ([`d744f71`](https://github.com/giacomoguidotto/anypinn/commit/d744f71d8a7c10edde68b2dcbf1b89180f2cfa6d))

- Replace fact blueprint template with anypinn user guide
  ([`1ec4fe9`](https://github.com/giacomoguidotto/anypinn/commit/1ec4fe9200021464ce382658b7ed067069ac8b99))

The generated docs still contained the python-blueprint's fact example. Rewrites index.md with
  actual anypinn content (installation, examples, architecture, problem definition guide), fixes
  mkdocs edit_uri, and removes stale src/fact/cli.py reference from CLAUDE.md.

### Features

- Implement scaffolding cli
  ([`77f1bcf`](https://github.com/giacomoguidotto/anypinn/commit/77f1bcf1ca4c0a09992a1b18734e7098900ccb77))

Add a Typer-based CLI that bootstraps PINN projects with clack-style interactive prompts. Supports 6
  templates (SIR, SEIR, Damped Oscillator, Lotka-Volterra, Custom, Blank), synthetic/CSV data
  sources, and Lightning/core-only training modes. All options accept flags for non-interactive use.

- Move rich, typer-slim, simple-term-menu to package dependencies - Delete placeholder src/fact/
  package and its tests - Add cli module: app, prompts, renderer, types, and template system -
  Generated projects split into ode.py, config.py, train.py, data/ - 53 tests covering all 24
  template combinations and edge cases


## v0.1.0 (2026-02-09)

### Bug Fixes

- Ci tests
  ([`9cfdd3e`](https://github.com/giacomoguidotto/anypinn/commit/9cfdd3edb06c2ec317e8be90d866cb7f9433fa96))

- Convergence ([#7](https://github.com/giacomoguidotto/anypinn/pull/7),
  [`d094723`](https://github.com/giacomoguidotto/anypinn/commit/d094723346a4a6bf358fa579417f69dcc8af1bdd))

- Damped oscilator convergence
  ([`cbf7fc1`](https://github.com/giacomoguidotto/anypinn/commit/cbf7fc1e62b2dfac351958d77941b64dd3a0a165))

- Lv convergence
  ([`50df3cf`](https://github.com/giacomoguidotto/anypinn/commit/50df3cfd04aa6e45f746182ed3e6751ad08cb340))

### Chores

- Add claude code support
  ([`7a34903`](https://github.com/giacomoguidotto/anypinn/commit/7a349036f91740c331382bff8f9b0dd908c07d6b))

- Different runs
  ([`3087f19`](https://github.com/giacomoguidotto/anypinn/commit/3087f1912df0e0888ec93535594af7e3816a924e))

- Sync with blueprint
  ([`b862fb3`](https://github.com/giacomoguidotto/anypinn/commit/b862fb3888a6dd5037b0282de39f8fd4314d114f))

- Sync with blueprint
  ([`ed436a5`](https://github.com/giacomoguidotto/anypinn/commit/ed436a5650445a2ff6c56c458c4748c7b1dc8349))

- Without scaling hospitalized run
  ([`76e38e5`](https://github.com/giacomoguidotto/anypinn/commit/76e38e5e356a6ef7c2400d310e4c94b7c60f7a65))

- **release**: V0.1.0
  ([`59133b4`](https://github.com/giacomoguidotto/anypinn/commit/59133b48c93a882ffec7a0e215ceb642bf6388b4))

### Continuous Integration

- Add automatic PyPI publishing with semantic-release
  ([`5796062`](https://github.com/giacomoguidotto/anypinn/commit/57960625615497c437503b74aaad3d33c3343b9b))

Configure python-semantic-release to auto-version from conventional commits and publish to PyPI via
  OIDC trusted publisher on each release.

### Documentation

- Replace ASCII diagrams with Mermaid in README
  ([`42f758b`](https://github.com/giacomoguidotto/anypinn/commit/42f758b56f3f538439a77603922c5359f95c7bb5))

- Update README
  ([`3e00e95`](https://github.com/giacomoguidotto/anypinn/commit/3e00e9529486b6539dd7ae59ed86c24ec7271b27))

### Features

- Add initial docs ([#6](https://github.com/giacomoguidotto/anypinn/pull/6),
  [`3f5af1f`](https://github.com/giacomoguidotto/anypinn/commit/3f5af1f892968ea81493108ed546583022eb6c6a))

- Context arch ([#8](https://github.com/giacomoguidotto/anypinn/pull/8),
  [`0f8b170`](https://github.com/giacomoguidotto/anypinn/commit/0f8b170c37136a835887e9c46501620d0f552e88))

- Core ([#2](https://github.com/giacomoguidotto/anypinn/pull/2),
  [`9818cb2`](https://github.com/giacomoguidotto/anypinn/commit/9818cb20743be6c04c67f036b2b9fcb8326edd5a))

- Hospitalized ([#9](https://github.com/giacomoguidotto/anypinn/pull/9),
  [`83b81de`](https://github.com/giacomoguidotto/anypinn/commit/83b81de027354a2e93976a8ae7e449a4e615cb26))

- Initialize pinn lib from original thesis ([#1](https://github.com/giacomoguidotto/anypinn/pull/1),
  [`d45449e`](https://github.com/giacomoguidotto/anypinn/commit/d45449e33a92265411edcfe77d679a77b70a006c))

- Ode generalization ([#4](https://github.com/giacomoguidotto/anypinn/pull/4),
  [`387aaee`](https://github.com/giacomoguidotto/anypinn/commit/387aaeeee0d7a76cc835c681de27fb949266486a))

- Optimizations ([#3](https://github.com/giacomoguidotto/anypinn/pull/3),
  [`f34a6f1`](https://github.com/giacomoguidotto/anypinn/commit/f34a6f15aba419ed6ee37172b30c96235b6e5656))

- Other examples
  ([`7915c1d`](https://github.com/giacomoguidotto/anypinn/commit/7915c1d71e5c33325184ecb394f598ed7e2d5a05))

- Save on interrupt
  ([`8ad9eee`](https://github.com/giacomoguidotto/anypinn/commit/8ad9eee03fee07d4eee5aee8ba6d83d401510e95))

- Sync with 'blueprint'
  ([`6f44b86`](https://github.com/giacomoguidotto/anypinn/commit/6f44b865dcfdcf2045a24a6f0fb7fcadf30f0c1a))

### Refactoring

- Create catalog module and generalize ODE abstractions
  ([`486e713`](https://github.com/giacomoguidotto/anypinn/commit/486e7137ba5f29ec2a46dff487589be7009008f6))

Move problem-specific building blocks (ODE functions, DataModules, constants) from sir_inverse.py
  and example scripts into a new pinn.catalog package. Generalize SIRInvHyperparameters to
  ODEHyperparameters and unify all identical Problem subclasses into a single ODEInverseProblem
  class in pinn.problems.ode.

- Rename package from pinn to anypinn
  ([`161aa1c`](https://github.com/giacomoguidotto/anypinn/commit/161aa1c68b0c4b4a45171333d523ea2a5c723fc1))

Rebrand the Python package: rename src/pinn/ to src/anypinn/, update all imports, config files,
  documentation, and serialized log artifacts. Class names like PINNModule remain unchanged as they
  refer to the technique.
