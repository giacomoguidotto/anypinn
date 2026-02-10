"""Clack-style interactive prompts using Rich + simple-term-menu."""

from __future__ import annotations

import sys
from typing import TypeVar

from rich.console import Console
from simple_term_menu import TerminalMenu

from anypinn.cli._types import DataSource, Template

_console = Console()

T = TypeVar("T")


def _print_bar() -> None:
    _console.print("[dim]│[/]")


def _clear_lines(n: int) -> None:
    """Move cursor up *n* lines and clear to end of screen."""
    sys.stdout.write(f"\033[{n}A\033[J")
    sys.stdout.flush()


def _select(question: str, options: list[T], labels: list[str]) -> T:
    """Display a clack-style selection prompt and return the chosen option."""
    _console.print(f"[bold cyan]◆[/]  {question}")
    _print_bar()

    menu = TerminalMenu(
        labels,
        menu_cursor="│  ● ",
        menu_cursor_style=("fg_cyan", "bold"),
        menu_highlight_style=("fg_cyan",),
    )
    raw_index = menu.show()

    if raw_index is None:
        raise SystemExit(1)

    index: int = int(raw_index)
    selected = options[index]

    # Overwrite the ◆ question + │ bar that stayed on screen
    _clear_lines(2)

    _console.print(f"[bold green]◇[/]  {question}")
    for i, lbl in enumerate(labels):
        if i == index:
            _console.print(f"[dim]│[/]  [bold green]●[/] {lbl}")
        else:
            _console.print(f"[dim]│[/]    [dim s]{lbl}[/]")
    _print_bar()

    return selected


def _confirm(question: str, default: bool = True) -> bool:
    """Display a clack-style yes/no prompt."""
    _console.print(f"[bold cyan]◆[/]  {question}")
    _print_bar()

    suffix = " [Y/n] " if default else " [y/N] "
    _console.print("[dim]│[/]  ", end="")
    answer = input(suffix).strip().lower()

    result = default if answer == "" else answer in ("y", "yes")

    display = "Yes" if result else "No"

    # Overwrite the ◆ question + │ bar + │ [Y/n] input line
    _clear_lines(3)

    _console.print(f"[bold green]◇[/]  {question}")
    _console.print(f"[dim]│[/]  {display}")
    _print_bar()

    return result


def prompt_template() -> Template:
    """Prompt user to choose a project template."""
    templates = list(Template)
    labels = [t.label for t in templates]
    return _select("Choose a starting point", templates, labels)


def prompt_data_source() -> DataSource:
    """Prompt user to choose a data source."""
    sources = list(DataSource)
    labels = [s.label for s in sources]
    return _select("Select training data source", sources, labels)


def prompt_lightning() -> bool:
    """Prompt user whether to include Lightning wrapper."""
    return _confirm("Include Lightning training wrapper?", default=True)
