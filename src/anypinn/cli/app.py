"""Typer CLI application for anypinn."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from typing import Annotated

from rich.console import Console
from typer import Argument, Exit, Option, Typer

import anypinn
from anypinn.cli._prompts import _confirm, prompt_data_source, prompt_lightning, prompt_template
from anypinn.cli._renderer import render_project
from anypinn.cli._types import DataSource, Template

app = Typer(add_completion=False, context_settings={"help_option_names": ["-h", "--help"]})
_console = Console(highlight=False)


def _version_callback(value: bool) -> None:
    if value:
        _console.print(f"anypinn {anypinn.__version__}")
        raise Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        Option(
            "--version",
            "-v",
            help="Show version and exit.",
            callback=_version_callback,
            is_eager=True,
            expose_value=False,
        ),
    ] = False,
) -> None:
    """anypinn - scaffolding tool for Physics-Informed Neural Network projects."""


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
    project_name: Annotated[str, Argument(help="Name for the new project directory")] = ".",
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
    run: Annotated[
        bool,
        Option("--run/--no-run", help="Install dependencies and run training after scaffolding"),
    ] = True,
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
    project_dir = Path(project_name).resolve()
    display_name = project_dir.name
    use_cwd = project_name == "."

    # Validate template early (non-interactive fast fail)
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

    # Handle existing directory
    if project_dir.exists():
        contents = list(project_dir.iterdir())
        if contents:
            if not _confirm(
                f"Directory '{display_name}' is not empty. Delete all contents?",
                default=False,
            ):
                _console.print()
                raise Exit(code=1)
            if not _confirm(
                "Are you sure? This cannot be undone",
                default=False,
            ):
                _console.print()
                raise Exit(code=1)
            for item in contents:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()

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
    _console.print(f"[bold green]◇[/]  Generating project in '{display_name}'")

    created = render_project(project_dir, template, data_source, lightning)

    for i, name in enumerate(created):
        desc = _FILE_DESCRIPTIONS.get(name, "")
        desc_str = f"\t\t[dim]{desc}[/]" if desc else ""
        connector = "└" if i == len(created) - 1 else "├"
        _console.print(f"[dim]│[/]  {connector} {name}{desc_str}")

    _console.print("[dim]│[/]")

    uv_path = shutil.which("uv") if run else None
    if uv_path is None:
        if use_cwd:
            _console.print("[bold cyan]●[/]  Done! uv sync && uv run train.py")
        else:
            _console.print(
                f"[bold cyan]●[/]  Done! cd {display_name} && uv sync && uv run train.py"
            )
        _console.print()
        return

    with _console.status(" Syncing dependencies...", spinner="dots", spinner_style="bold cyan"):
        proc = subprocess.run([uv_path, "sync"], capture_output=True, text=True, cwd=project_dir)

    if proc.returncode != 0:
        _console.print("[bold red]✗[/]  Failed to sync dependencies")
        if proc.stderr:
            _console.print(proc.stderr.strip())
        raise Exit(code=1)

    _console.print("[bold green]◇[/]  Dependencies synced")
    _console.print("[dim]│[/]")
    _console.print("[bold cyan]●[/]  Running `uv run train.py`...")
    _console.print()

    os.chdir(project_dir)
    os.execvp(uv_path, [uv_path, "run", "train.py"])
