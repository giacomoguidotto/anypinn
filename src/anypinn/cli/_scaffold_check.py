"""Check consistency between scaffold canonical files and examples.

Compares key constants and ODE function signatures between scaffold
templates and their corresponding examples. Run via ``just check-scaffold-match``.
"""

from __future__ import annotations

import ast
from pathlib import Path
import re
import sys

# Map scaffold directory to example directory
_SCAFFOLD_TO_EXAMPLE: dict[str, str] = {
    "sir": "sir_inverse",
    "seir": "seir_inverse",
    "damped_oscillator": "damped_oscillator",
    "lotka_volterra": "lotka_volterra",
    "lorenz": "lorenz",
    "van_der_pol": "van_der_pol",
    "fitzhugh_nagumo": "fitzhugh_nagumo",
    "gray_scott_2d": "gray_scott_2d",
    "poisson_2d": "poisson_2d",
    "heat_1d": "heat_1d",
    "burgers_1d": "burgers_1d",
    "wave_1d": "wave_1d",
    "inverse_diffusivity": "inverse_diffusivity",
    "allen_cahn": "allen_cahn",
}

_ROOT = Path(__file__).resolve().parents[3]
_SCAFFOLD_DIR = _ROOT / "src" / "anypinn" / "cli" / "scaffold"
_EXAMPLE_DIR = _ROOT / "examples"


def _extract_keys(source: str) -> set[str]:
    """Extract key constant names (ending in _KEY) from assignments and imports."""
    keys: set[str] = set()
    # Local assignments: BETA_KEY = "beta"
    for match in re.finditer(r"^([A-Z][A-Z0-9_]*_KEY)\s*=", source, re.MULTILINE):
        keys.add(match.group(1))
    # Any _KEY reference in the source (covers both single/multiline imports and usage)
    for match in re.finditer(r"\b([A-Z][A-Z0-9_]*_KEY)\b", source):
        keys.add(match.group(1))
    return keys


def _extract_function_names(source: str) -> set[str]:
    """Extract top-level function names from source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def check_pair(scaffold_name: str, example_name: str) -> list[str]:
    """Check consistency between a scaffold and its example. Returns issues."""
    issues: list[str] = []

    scaffold_ode = _SCAFFOLD_DIR / scaffold_name / "ode.py"
    example_ode = _EXAMPLE_DIR / example_name / "ode.py"

    if not scaffold_ode.exists():
        issues.append(f"  Missing scaffold: {scaffold_ode.relative_to(_ROOT)}")
        return issues
    if not example_ode.exists():
        issues.append(f"  Missing example: {example_ode.relative_to(_ROOT)}")
        return issues

    scaffold_src = scaffold_ode.read_text()
    example_src = example_ode.read_text()

    # Check that key constants (ending in _KEY) exist in both
    scaffold_keys = _extract_keys(scaffold_src)
    example_keys = _extract_keys(example_src)

    # Scaffold is a superset (may have keys for inverse variants not in forward-only examples).
    # Only flag keys that are in the example but missing from the scaffold.
    missing_in_scaffold = example_keys - scaffold_keys
    if missing_in_scaffold:
        issues.append(f"  Keys in example but not scaffold: {missing_in_scaffold}")

    # Check that create_problem exists in both
    scaffold_fns = _extract_function_names(scaffold_src)
    example_fns = _extract_function_names(example_src)

    if "create_problem" not in example_fns:
        # Check for suffixed variants
        has_problem = any("create_problem" in fn for fn in example_fns)
        if not has_problem:
            issues.append("  Missing create_problem in example")

    if "create_problem" not in scaffold_fns:
        # Scaffold may have suffixed variants (create_problem_forward, etc.)
        has_problem = any("create_problem" in fn for fn in scaffold_fns)
        if not has_problem:
            issues.append("  Missing create_problem in scaffold")

    return issues


def main() -> None:
    """Run all scaffold-example consistency checks."""
    all_ok = True

    for scaffold_name, example_name in _SCAFFOLD_TO_EXAMPLE.items():
        issues = check_pair(scaffold_name, example_name)
        if issues:
            print(f"MISMATCH: {scaffold_name} <-> {example_name}")
            for issue in issues:
                print(issue)
            all_ok = False

    if all_ok:
        print(f"All {len(_SCAFFOLD_TO_EXAMPLE)} scaffold-example pairs are consistent.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
