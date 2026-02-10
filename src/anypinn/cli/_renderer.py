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

    for filename, content in files.items():
        (project_dir / filename).write_text(content)

    return [*files.keys(), "data/"]
