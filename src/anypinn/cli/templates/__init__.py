"""Template modules for project scaffolding."""

from __future__ import annotations

from typing import Protocol

from anypinn.cli._types import DataSource


class TemplateModule(Protocol):
    """Protocol for template modules. Each exposes a render() returning filename -> content."""

    def render(self, data_source: DataSource, lightning: bool) -> dict[str, str]: ...
