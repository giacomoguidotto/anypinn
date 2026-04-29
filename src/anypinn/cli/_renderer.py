"""Orchestrates template rendering to files on disk."""

from __future__ import annotations

import importlib.resources as ilr
from pathlib import Path

from anypinn.cli._generator import extract_variants
from anypinn.cli._types import DataSource, Template

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


def _read(pkg: str, filename: str, experiment_name: str) -> str:
    content = ilr.files(pkg).joinpath(filename).read_text(encoding="utf-8")
    return content.replace("__EXPERIMENT_NAME__", experiment_name)


def _has_canonical(pkg: str, filename: str) -> bool:
    """Check if a canonical source file exists in the package."""
    return ilr.files(pkg).joinpath(filename).is_file()


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
) -> list[str]:
    """Render a template to files on disk. Returns list of created file/dir names."""
    tdir = _TEMPLATE_DIRS[template]
    exp = _EXPERIMENT_NAMES[template]
    ds = "synthetic" if data_source == DataSource.SYNTHETIC else "csv"
    tr = "lightning" if lightning else "core"
    pkg = f"anypinn.cli.scaffold.{tdir}"
    selections = {"source": ds}

    # Use canonical files (ode.py, config.py) with variant extraction when
    # available, falling back to legacy per-variant files (ode_{ds}.py).
    if _has_canonical(pkg, "ode.py"):
        ode = _read_canonical(pkg, "ode.py", exp, selections)
    else:
        ode = _read(pkg, f"ode_{ds}.py", exp)

    if _has_canonical(pkg, "config.py"):
        config = _read_canonical(pkg, "config.py", exp, selections)
    else:
        config = _read(pkg, f"config_{ds}.py", exp)

    files = {
        "ode.py": ode,
        "config.py": config,
        "train.py": _read("anypinn.cli.scaffold._shared", f"train_{tr}.py", exp),
    }

    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "data").mkdir()

    pyproject = _pyproject_toml(project_dir.name, data_source, lightning)
    (project_dir / "pyproject.toml").write_text(pyproject)

    for name, content in files.items():
        (project_dir / name).write_text(content)

    return ["pyproject.toml", *files.keys(), "data/"]
