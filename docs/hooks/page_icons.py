"""Inject page icons for MkDocs without Markdown front matter.

Keeping icon metadata in this hook avoids YAML front matter showing up in
GitHub's Markdown renderer, while still exposing ``page.meta["icon"]`` to
Material for MkDocs features.
"""

from __future__ import annotations

PAGE_ICONS: dict[str, str] = {
    "index.md": "material/math-integral-box",
    "getting-started/index.md": "material/rocket-launch",
    "getting-started/pinn-primer.md": "material/school",
    "getting-started/installation.md": "material/download",
    "getting-started/first-project.md": "material/play-circle",
    "getting-started/understanding-the-output.md": "material/chart-line",
    "getting-started/next-steps.md": "material/lightbulb",
    "guides/index.md": "material/book-open-variant",
    "guides/custom-ode.md": "material/function-variant",
    "guides/promote-constant-to-parameter.md": "material/swap-horizontal",
    "guides/inverse-vs-forward.md": "material/arrow-split-vertical",
    "guides/pde-forward-problems.md": "material/grid",
    "guides/csv-data.md": "material/file-delimited",
    "guides/tune-hyperparameters.md": "material/tune-vertical",
    "guides/loss-weighting.md": "material/scale-balance",
    "guides/lightning-vs-core.md": "material/lightning-bolt",
    "guides/architecture.md": "material/layers-triple",
    "catalog/index.md": "material/file-document-multiple",
    "reference/index.md": "material/api",
    "reference/core.md": "material/cube-outline",
    "reference/problems.md": "material/function-variant",
    "reference/lightning.md": "material/lightning-bolt",
    "reference/cli.md": "material/console",
    "CONTRIBUTING.md": "material/source-pull",
    "CODE_OF_CONDUCT.md": "material/handshake",
}


def on_page_markdown(markdown: str, page, **_: object) -> str:
    """Assign icon metadata by source path before Markdown rendering."""
    if icon := PAGE_ICONS.get(page.file.src_uri):
        page.meta["icon"] = icon
    return markdown
