"""Rewrite ``docs/`` relative links for the symlinked index page.

``docs/index.md`` is a symlink to ``../README.md``.  Links like
``docs/getting-started.md`` work on GitHub (relative to the repo root) but
break inside MkDocs (which resolves them from the ``docs/`` directory).  This
hook strips the ``docs/`` prefix so both contexts resolve correctly.
"""

from __future__ import annotations

import re


def on_page_markdown(markdown: str, page, **_: object) -> str:
    """Strip ``docs/`` prefix from Markdown links on the index page."""
    if page.file.src_uri != "index.md":
        return markdown
    return re.sub(r"\]\(docs/", "](", markdown)
