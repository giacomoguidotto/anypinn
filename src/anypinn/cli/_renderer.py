"""Orchestrates template rendering to files on disk."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

from anypinn.cli._types import DataSource, Template
from anypinn.cli.templates import _blank as blank
from anypinn.cli.templates import _custom as custom
from anypinn.cli.templates import _damped_oscillator as damped_oscillator
from anypinn.cli.templates import _lotka_volterra as lotka_volterra
from anypinn.cli.templates import _seir as seir
from anypinn.cli.templates import _sir as sir

RenderFn = Callable[[DataSource, bool], dict[str, str]]

_TEMPLATES: dict[Template, ModuleType] = {
    Template.SIR: sir,
    Template.SEIR: seir,
    Template.DAMPED_OSCILLATOR: damped_oscillator,
    Template.LOTKA_VOLTERRA: lotka_volterra,
    Template.CUSTOM: custom,
    Template.BLANK: blank,
}

_BASE_DEPS: list[str] = [
    "anypinn",
    "torch",
    "numpy",
    "scipy",
    "pandas",
]

_LIGHTNING_DEPS: list[str] = [
    "lightning",
    "tensorboard",
]

_SYNTHETIC_DEPS: list[str] = [
    "torchdiffeq",
]


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
    mod = _TEMPLATES[template]
    render_fn: RenderFn = mod.render
    files = render_fn(data_source, lightning)

    project_dir.mkdir(parents=True)
    (project_dir / "data").mkdir()

    pyproject = _pyproject_toml(project_dir.name, data_source, lightning)
    (project_dir / "pyproject.toml").write_text(pyproject)

    for filename, content in files.items():
        (project_dir / filename).write_text(content)

    return ["pyproject.toml", *files.keys(), "data/"]
