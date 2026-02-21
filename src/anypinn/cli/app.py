"""Typer CLI application for anypinn."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from rich.console import Console
from typer import Argument, Exit, Option, Typer

import anypinn
from anypinn.cli._prompts import prompt_data_source, prompt_lightning, prompt_template
from anypinn.cli._renderer import render_project
from anypinn.cli._types import DataSource, Template

app = Typer(add_completion=False, context_settings={"help_option_names": ["-h", "--help"]})
_console = Console()


@app.callback()
def main() -> None:
    """anypinn — scaffolding tool for Physics-Informed Neural Network projects."""


_FILE_DESCRIPTIONS: dict[str, str] = {
    "pyproject.toml": "project dependencies",
    "ode.py": "mathematical definition",
    "config.py": "training configuration",
    "train.py": "execution script",
    "data/": "data directory",
}


def _print_templates() -> None:
    _console.print()
    _console.print("[bold cyan]◆[/]  Available templates")
    _console.print("[dim]│[/]")
    for t in Template:
        _console.print(f"[dim]│[/]  [bold cyan]{t.value:<22}[/] [bold]{t.label}[/]")
        _console.print(f"[dim]│[/]  {' ' * 22} [dim]{t.description}[/]")
        _console.print("[dim]│[/]")
    _console.print()


def _list_templates_callback(value: bool) -> None:
    if value:
        _print_templates()
        raise Exit()


@app.command()
def create(
    project_name: Annotated[str, Argument(help="Name for the new project directory")],
    template_str: Annotated[
        str | None,
        Option(
            "--template",
            "-t",
            help="Project template. Run with --list-templates / -l to see all options.",
            show_default=False,
        ),
    ] = None,
    data_source: Annotated[
        DataSource | None, Option("--data", "-d", help="Training data source")
    ] = None,
    lightning: Annotated[
        bool | None,
        Option("--lightning/--no-lightning", "-L/-NL", help="Include Lightning wrapper"),
    ] = None,
    list_templates: Annotated[
        bool,
        Option(
            "--list-templates",
            "-l",
            help="List all available templates and exit.",
            callback=_list_templates_callback,
            is_eager=True,
            expose_value=False,
        ),
    ] = False,
) -> None:
    """Create a new PINN project."""
    project_dir = Path(project_name)

    if project_dir.exists():
        _console.print(f"[bold red]Error:[/] Directory '{project_name}' already exists.")
        raise Exit(code=1)

    template: Template | None = None
    if template_str is not None:
        try:
            template = Template(template_str)
        except ValueError:
            valid = ", ".join(f"'{t.value}'" for t in Template)
            _console.print()
            _console.print(
                f"[bold red]Error:[/] [bold]{template_str!r}[/] is not a valid template."
            )
            _console.print(f"[dim]Valid values:[/] {valid}")
            _print_templates()
            raise Exit(code=2) from None

    # Header
    _console.print()
    _console.print(f"[bold cyan]●[/]  anypinn v{anypinn.__version__}")
    _console.print("[dim]│[/]")

    # Interactive prompts for missing options
    if template is None:
        template = prompt_template()
    else:
        _console.print("[bold green]◇[/]  Choose a starting point")
        _console.print(f"[dim]│[/]  {template.label}")
        _console.print("[dim]│[/]")

    if data_source is None:
        data_source = prompt_data_source()
    else:
        _console.print("[bold green]◇[/]  Select training data source")
        _console.print(f"[dim]│[/]  {data_source.label}")
        _console.print("[dim]│[/]")

    if lightning is None:
        lightning = prompt_lightning()
    else:
        display = "Yes" if lightning else "No"
        _console.print("[bold green]◇[/]  Include Lightning training wrapper?")
        _console.print(f"[dim]│[/]  {display}")
        _console.print("[dim]│[/]")

    # Render
    _console.print(f"[bold green]◇[/]  Creating {project_name}/...")

    created = render_project(project_dir, template, data_source, lightning)

    for name in created:
        desc = _FILE_DESCRIPTIONS.get(name, "")
        desc_str = f" [dim]— {desc}[/]" if desc else ""
        _console.print(f"[dim]│[/]  {name}{desc_str}")

    _console.print("[dim]│[/]")
    _console.print(f"[bold cyan]●[/]  Done! cd {project_name} && uv sync && uv run train.py")
    _console.print()
