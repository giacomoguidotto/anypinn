"""Variant extraction from canonical scaffold source files.

Canonical source files contain all variant code (uncommented, lintable Python)
with marker comments delimiting variant-specific blocks. Each block is tagged
with an axis and value, and names within the block carry a ``_<value>`` suffix.

Marker syntax::

    # --- VARIANT: source/synthetic ---
    validation_synthetic: ValidationRegistry = { ... }
    # --- VARIANT: source/csv ---
    validation_csv: ValidationRegistry = { ... }
    # --- END VARIANT ---

The generator selects one value per axis and:
  1. Keeps only the selected blocks.
  2. Strips the ``_<value>`` suffix from identifiers.
  3. Removes the marker comments themselves.
"""

from __future__ import annotations

import re

# Matches: # --- VARIANT: axis/value ---
_VARIANT_START = re.compile(r"^(\s*)# --- VARIANT: (\w+)/(\w+) ---\s*$")
# Matches: # --- END VARIANT ---
_VARIANT_END = re.compile(r"^\s*# --- END VARIANT ---\s*$")


def extract_variants(source: str, selections: dict[str, str]) -> str:
    """Extract selected variants from a canonical source file.

    Args:
        source: Full contents of a canonical source file with variant markers.
        selections: Mapping of axis name to selected value,
            e.g. ``{"source": "synthetic", "direction": "inverse"}``.

    Returns:
        Processed source with only selected variant blocks retained,
        suffixes stripped, and marker comments removed.
    """
    lines = source.splitlines(keepends=True)
    result: list[str] = []

    # Stack tracks (axis, value, keep) for nested variants (though nesting
    # is unlikely, the logic is cleaner with a stack).
    stack: list[tuple[str, str, bool]] = []

    for line in lines:
        start_match = _VARIANT_START.match(line)
        end_match = _VARIANT_END.match(line)

        if start_match:
            _, axis, value = start_match.groups()
            selected = selections.get(axis)
            keep = selected == value
            # If the top of the stack is the same axis, this is an
            # alternative variant in the same group — replace it.
            if stack and stack[-1][0] == axis:
                stack[-1] = (axis, value, keep)
            else:
                stack.append((axis, value, keep))
            continue

        if end_match:
            if stack:
                stack.pop()
            continue

        # If inside a variant block, only keep lines from selected variants.
        if stack:
            _, _, keep = stack[-1]
            if not keep:
                continue

        result.append(line)

    output = "".join(result)

    # Strip suffixes: for each selected value, remove the _<value> suffix
    # from identifiers. We match word boundaries to avoid false replacements.
    for value in selections.values():
        output = re.sub(rf"(\w)_{value}\b", r"\1", output)

    # Collapse runs of 3+ newlines down to 2 (one blank line).
    output = re.sub(r"\n{3,}", "\n\n", output)

    # Remove unused imports left over from variant extraction.
    output = _remove_unused_imports(output)

    return output


# Patterns for import statements
_IMPORT_SIMPLE = re.compile(r"^import (\w+)")
_IMPORT_FROM = re.compile(r"^from [\w.]+ import (.+)$")


def _remove_unused_imports(source: str) -> str:
    """Remove import names that are not referenced in the rest of the file."""
    lines = source.split("\n")
    non_import_text = "\n".join(
        line for line in lines if not line.startswith(("import ", "from "))
    )

    result: list[str] = []
    for line in lines:
        m_simple = _IMPORT_SIMPLE.match(line)
        if m_simple:
            name = m_simple.group(1)
            if name not in ("__future__",) and name not in non_import_text:
                continue
            result.append(line)
            continue

        m_from = _IMPORT_FROM.match(line)
        if m_from:
            # Always keep __future__ imports
            if line.startswith("from __future__"):
                result.append(line)
                continue
            names_str = m_from.group(1).strip()
            # Skip complex imports (multiline, already handled by inline markers)
            if names_str.startswith("(") or line.endswith("\\"):
                result.append(line)
                continue
            names = [n.strip() for n in names_str.split(",") if n.strip()]
            used = [n for n in names if n in non_import_text]
            if not used:
                continue
            if len(used) < len(names):
                # Reconstruct import with only used names
                module = line.split(" import ")[0]
                line = f"{module} import {', '.join(used)}"
            result.append(line)
            continue

        result.append(line)

    return "\n".join(result)
