"""Orchestrates template rendering to files on disk."""

from __future__ import annotations

import importlib.resources as ilr
from pathlib import Path

from anypinn.cli._generator import extract_variants
from anypinn.cli._types import DataSource, Direction, Template

_TEMPLATE_DIRS: dict[Template, str] = {
    Template.SIR: "sir",
    Template.SEIR: "seir",
    Template.DAMPED_OSCILLATOR: "damped_oscillator",
    Template.LOTKA_VOLTERRA: "lotka_volterra",
    Template.VAN_DER_POL: "van_der_pol",
    Template.LORENZ: "lorenz",
    Template.FITZHUGH_NAGUMO: "fitzhugh_nagumo",
    Template.GRAY_SCOTT_2D: "gray_scott_2d",
    Template.POISSON_2D: "poisson_2d",
    Template.HEAT_1D: "heat_1d",
    Template.BURGERS_1D: "burgers_1d",
    Template.WAVE_1D: "wave_1d",
    Template.INVERSE_DIFFUSIVITY: "inverse_diffusivity",
    Template.ALLEN_CAHN: "allen_cahn",
    Template.CUSTOM: "custom",
    Template.BLANK: "blank",
}

_EXPERIMENT_NAMES: dict[Template, str] = {
    Template.SIR: "sir-inverse",
    Template.SEIR: "seir-inverse",
    Template.DAMPED_OSCILLATOR: "damped-oscillator",
    Template.LOTKA_VOLTERRA: "lotka-volterra",
    Template.VAN_DER_POL: "van-der-pol",
    Template.LORENZ: "lorenz",
    Template.FITZHUGH_NAGUMO: "fitzhugh-nagumo",
    Template.GRAY_SCOTT_2D: "gray-scott-2d",
    Template.POISSON_2D: "poisson-2d",
    Template.HEAT_1D: "heat-1d",
    Template.BURGERS_1D: "burgers-1d",
    Template.WAVE_1D: "wave-1d",
    Template.INVERSE_DIFFUSIVITY: "inverse-diffusivity",
    Template.ALLEN_CAHN: "allen-cahn",
    Template.CUSTOM: "custom-ode",
    Template.BLANK: "my-project",
}

_BASE_DEPS: list[str] = [
    "anypinn",
    "numpy",
    "scipy",
]

_LIGHTNING_DEPS: list[str] = [
    "tensorboard",
]

_SYNTHETIC_DEPS: list[str] = []

# Stub CSV column definitions per template (x_column + y_columns).
# Used to generate placeholder data so the CSV variant runs out of the box.
_CSV_COLUMNS: dict[Template, tuple[str, list[str], int]] = {
    # (x_column, y_columns, n_rows)
    Template.SIR: ("t", ["I_obs"], 91),
    Template.SEIR: ("t", ["I_obs"], 161),
    Template.DAMPED_OSCILLATOR: ("t", ["x_obs"], 200),
    Template.LOTKA_VOLTERRA: ("t", ["x_obs", "y_obs"], 200),
    Template.VAN_DER_POL: ("t", ["u_obs"], 200),
    Template.LORENZ: ("t", ["x_obs", "y_obs", "z_obs"], 200),
    Template.FITZHUGH_NAGUMO: ("t", ["v"], 200),
    Template.GRAY_SCOTT_2D: ("t", ["u", "v"], 100),
    Template.POISSON_2D: ("x", ["u"], 100),
    Template.HEAT_1D: ("t", ["u"], 100),
    Template.BURGERS_1D: ("t", ["u"], 100),
    Template.WAVE_1D: ("t", ["u"], 100),
    Template.INVERSE_DIFFUSIVITY: ("t", ["u"], 100),
    Template.ALLEN_CAHN: ("t", ["u"], 100),
    Template.CUSTOM: ("x", ["y_obs"], 50),
    Template.BLANK: ("x", ["y_obs"], 50),
}


def _stub_csv(template: Template) -> str:
    """Generate a stub CSV file with placeholder data."""
    import random

    x_col, y_cols, n = _CSV_COLUMNS.get(template, ("x", ["y"], 50))
    header = ",".join([x_col, *y_cols])
    lines = [header]
    for i in range(n):
        x_val = f"{i / (n - 1):.6f}" if n > 1 else "0.000000"
        y_vals = ",".join(f"{random.uniform(0, 1):.6f}" for _ in y_cols)
        lines.append(f"{x_val},{y_vals}")
    return "\n".join(lines) + "\n"


def _read(pkg: str, filename: str, experiment_name: str) -> str:
    content = ilr.files(pkg).joinpath(filename).read_text(encoding="utf-8")
    return content.replace("__EXPERIMENT_NAME__", experiment_name)


def _read_canonical(
    pkg: str, filename: str, experiment_name: str, selections: dict[str, str]
) -> str:
    """Read a canonical source file and extract selected variants."""
    content = ilr.files(pkg).joinpath(filename).read_text(encoding="utf-8")
    content = content.replace("__EXPERIMENT_NAME__", experiment_name)
    return extract_variants(content, selections)


def _pyproject_toml(project_name: str, data_source: DataSource, lightning: bool) -> str:
    """Generate a minimal pyproject.toml for the scaffolded project."""
    deps = (
        _BASE_DEPS
        + (_SYNTHETIC_DEPS if data_source == DataSource.SYNTHETIC else [])
        + (_LIGHTNING_DEPS if lightning else [])
    )
    deps_str = "\n".join(f'    "{d}",' for d in deps)

    return f"""\
[project]
name = "{project_name}"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
{deps_str}
]

[tool.uv]
package = false
"""


def render_project(
    project_dir: Path,
    template: Template,
    data_source: DataSource,
    lightning: bool,
    direction: Direction | None = None,
) -> list[str]:
    """Render a template to files on disk. Returns list of created file/dir names."""
    tdir = _TEMPLATE_DIRS[template]
    exp = _EXPERIMENT_NAMES[template]
    ds = "synthetic" if data_source == DataSource.SYNTHETIC else "csv"
    tr = "lightning" if lightning else "core"
    pkg = f"anypinn.cli.scaffold.{tdir}"
    selections: dict[str, str] = {"source": ds}
    if direction is not None:
        selections["direction"] = direction.value

    files = {
        "ode.py": _read_canonical(pkg, "ode.py", exp, selections),
        "config.py": _read_canonical(pkg, "config.py", exp, selections),
        "train.py": _read("anypinn.cli.scaffold._shared", f"train_{tr}.py", exp),
    }

    project_dir.mkdir(parents=True, exist_ok=True)
    data_dir = project_dir / "data"
    data_dir.mkdir()

    if data_source == DataSource.CSV:
        (data_dir / "data.csv").write_text(_stub_csv(template))

    pyproject = _pyproject_toml(project_dir.name, data_source, lightning)
    (project_dir / "pyproject.toml").write_text(pyproject)

    for name, content in files.items():
        (project_dir / name).write_text(content)

    return ["pyproject.toml", *files.keys(), "data/"]
