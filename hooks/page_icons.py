"""Inject page icons for MkDocs without Markdown front matter.

Keeping icon metadata in this hook avoids YAML front matter showing up in
GitHub's Markdown renderer, while still exposing ``page.meta["icon"]`` to
Material for MkDocs features.
"""

from __future__ import annotations

PAGE_ICONS: dict[str, str] = {
    "index.md": "material/math-integral-box",
    "CONTRIBUTING.md": "material/source-pull",
    "CODE_OF_CONDUCT.md": "material/handshake",
    "rationale.md": "material/scale-balance",
}


def on_page_markdown(markdown: str, page, **_: object) -> str:
    """Assign icon metadata by source path before Markdown rendering."""
    if icon := PAGE_ICONS.get(page.file.src_uri):
        page.meta["icon"] = icon
    return markdown
