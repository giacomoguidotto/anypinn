# Development task runner — replaces nox

# Default: list available recipes
default:
    @just --list

# Run tests with coverage
test *args:
    uv run python -m pytest --cov=anypinn --cov-report=html --cov-report=term tests {{ args }}

# Format code (isort + ruff)
fmt:
    uv run ruff check . --select I --select F401 --extend-fixable F401 --fix
    uv run ruff format .

# Check code style
lint:
    uv run ruff check .
    uv run ruff format --check .

# Auto-fix linting issues
lint-fix:
    uv run ruff check . --extend-fixable F401 --fix

# Type check
check:
    uv run ty check

# Build docs
docs:
    PYTHONPATH=src uv run python -m mkdocs build

# Build docs with external URL validation
docs-check-urls:
    PYTHONPATH=src HTMLPROOFER_VALIDATE_EXTERNAL_URLS=True uv run python -m mkdocs build

# Build docs for offline use
docs-offline:
    PYTHONPATH=src MKDOCS_MATERIAL_OFFLINE=True uv run python -m mkdocs build

# Serve docs locally
docs-serve:
    PYTHONPATH=src uv run python -m mkdocs serve

# Deploy docs to GitHub Pages
docs-deploy:
    PYTHONPATH=src uv run python -m mkdocs gh-deploy --force

# Generate license report
licenses *args:
    uv run --only-group licenses pip-licenses --format=markdown --output-file=./docs/licenses/summary.txt {{ args }}
    uv run --only-group licenses pip-licenses --format=plain-vertical --with-license-files --no-license-path --output-file=./docs/licenses/license_files.txt
    uv run --only-group licenses pip-licenses {{ args }}
    uv run --only-group licenses pip-licenses --summary

# Quick local CI check (lint + type check + test)
ci: lint check test
