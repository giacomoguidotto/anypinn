# Blank

```bash
anypinn create my-project --template blank
```

Empty project structure with no ODE definition. Generates only the scaffolding files
(`pyproject.toml`, `config.py`, `train.py`, `data/`) with no physics pre-configured.

## When to Use

Use this when you want the project layout and dependency management but plan to write everything
from scratch, or when working with a problem type not covered by `ODEInverseProblem` or the PDE
constraints.
