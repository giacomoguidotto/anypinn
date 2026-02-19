"""Integration tests for the anypinn CLI."""

from __future__ import annotations

from pathlib import Path

import click
import pytest
from typer.testing import CliRunner

from anypinn.cli import app
from anypinn.cli._types import DataSource, Template

runner = CliRunner()


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    return tmp_path / "test-project"


class TestCreateCommand:
    """Tests for the `create` command with non-interactive flags."""

    @pytest.mark.parametrize("template", list(Template))
    @pytest.mark.parametrize("data_source", list(DataSource))
    @pytest.mark.parametrize("lightning", [True, False])
    def test_all_combinations(
        self,
        tmp_path: Path,
        template: Template,
        data_source: DataSource,
        lightning: bool,
    ) -> None:
        """Every template x data_source x lightning combo generates valid Python."""
        name = f"proj-{template.value}-{data_source.value}-{lightning}"
        project = tmp_path / name

        lightning_flag = "--lightning" if lightning else "--no-lightning"
        result = runner.invoke(
            app,
            [
                "create",
                str(project),
                "--template",
                template.value,
                "--data",
                data_source.value,
                lightning_flag,
            ],
        )

        assert result.exit_code == 0, result.output

        # All templates produce pyproject.toml, ode.py, config.py, train.py, data/
        assert (project / "pyproject.toml").exists()
        assert (project / "ode.py").exists()
        assert (project / "config.py").exists()
        assert (project / "train.py").exists()
        assert (project / "data").is_dir()

        # Every generated .py must be valid syntax
        for py_file in project.glob("*.py"):
            source = py_file.read_text()
            compile(source, str(py_file), "exec")

    def test_existing_directory_fails(self, project_dir: Path) -> None:
        """Creating a project in an existing directory should fail."""
        project_dir.mkdir(parents=True)

        result = runner.invoke(
            app,
            [
                "create",
                str(project_dir),
                "--template",
                "sir",
                "--data",
                "synthetic",
                "--lightning",
            ],
        )

        assert result.exit_code == 1
        assert "already exists" in result.output

    def test_no_lightning_contamination(self, project_dir: Path) -> None:
        """Core-only train.py must not reference Lightning imports."""
        result = runner.invoke(
            app,
            [
                "create",
                str(project_dir),
                "--template",
                "sir",
                "--data",
                "synthetic",
                "--no-lightning",
            ],
        )

        assert result.exit_code == 0

        train_content = (project_dir / "train.py").read_text()
        assert "lightning" not in train_content.lower()
        assert "Trainer" not in train_content

    def test_lightning_includes_trainer(self, project_dir: Path) -> None:
        """Lightning train.py must reference Trainer."""
        result = runner.invoke(
            app,
            [
                "create",
                str(project_dir),
                "--template",
                "sir",
                "--data",
                "synthetic",
                "--lightning",
            ],
        )

        assert result.exit_code == 0

        train_content = (project_dir / "train.py").read_text()
        assert "Trainer" in train_content
        assert "PINNModule" in train_content

    def test_no_stray_format_braces(self, project_dir: Path) -> None:
        """Generated files should not contain stray { or } from f-string bugs."""
        result = runner.invoke(
            app,
            [
                "create",
                str(project_dir),
                "--template",
                "sir",
                "--data",
                "synthetic",
                "--lightning",
            ],
        )

        assert result.exit_code == 0

        for py_file in project_dir.glob("*.py"):
            source = py_file.read_text()
            # Valid Python should compile without errors
            compile(source, str(py_file), "exec")

    def test_output_shows_done(self, project_dir: Path) -> None:
        """CLI output includes the 'Done!' message."""
        result = runner.invoke(
            app,
            [
                "create",
                str(project_dir),
                "--template",
                "blank",
                "--data",
                "synthetic",
                "--lightning",
            ],
        )

        assert result.exit_code == 0
        assert "Done!" in result.output

    def test_csv_data_source_references_csv(self, project_dir: Path) -> None:
        """CSV data source should reference IngestionConfig in config.py."""
        result = runner.invoke(
            app,
            [
                "create",
                str(project_dir),
                "--template",
                "sir",
                "--data",
                "csv",
                "--lightning",
            ],
        )

        assert result.exit_code == 0

        config_content = (project_dir / "config.py").read_text()
        assert "IngestionConfig" in config_content
        assert "df_path" in config_content

    def test_pyproject_includes_lightning_deps(self, project_dir: Path) -> None:
        """Lightning project pyproject.toml must list lightning dependency."""
        result = runner.invoke(
            app,
            [
                "create",
                str(project_dir),
                "--template",
                "sir",
                "--data",
                "synthetic",
                "--lightning",
            ],
        )

        assert result.exit_code == 0

        pyproject = (project_dir / "pyproject.toml").read_text()
        assert '"lightning"' in pyproject
        assert '"tensorboard"' in pyproject

    def test_pyproject_excludes_lightning_deps(self, project_dir: Path) -> None:
        """Core-only project pyproject.toml must not list lightning dependency."""
        result = runner.invoke(
            app,
            [
                "create",
                str(project_dir),
                "--template",
                "sir",
                "--data",
                "synthetic",
                "--no-lightning",
            ],
        )

        assert result.exit_code == 0

        pyproject = (project_dir / "pyproject.toml").read_text()
        assert '"anypinn"' in pyproject
        assert '"torch"' in pyproject
        assert '"lightning"' not in pyproject

    def test_synthetic_data_source_references_generation(self, project_dir: Path) -> None:
        """Synthetic data source should reference GenerationConfig in config.py."""
        result = runner.invoke(
            app,
            [
                "create",
                str(project_dir),
                "--template",
                "sir",
                "--data",
                "synthetic",
                "--lightning",
            ],
        )

        assert result.exit_code == 0

        config_content = (project_dir / "config.py").read_text()
        assert "GenerationConfig" in config_content
        assert "linspace" in config_content

    def test_pyproject_includes_torchdiffeq_for_synthetic(self, project_dir: Path) -> None:
        """Synthetic data projects must include torchdiffeq dependency."""
        result = runner.invoke(
            app,
            [
                "create",
                str(project_dir),
                "--template",
                "sir",
                "--data",
                "synthetic",
                "--lightning",
            ],
        )

        assert result.exit_code == 0

        pyproject = (project_dir / "pyproject.toml").read_text()
        assert '"torchdiffeq"' in pyproject

    def test_pyproject_excludes_torchdiffeq_for_csv(self, project_dir: Path) -> None:
        """CSV data projects must not include torchdiffeq dependency."""
        result = runner.invoke(
            app,
            [
                "create",
                str(project_dir),
                "--template",
                "sir",
                "--data",
                "csv",
                "--lightning",
            ],
        )

        assert result.exit_code == 0

        pyproject = (project_dir / "pyproject.toml").read_text()
        assert '"torchdiffeq"' not in pyproject


class TestListTemplates:
    def test_list_templates_exits_zero(self) -> None:
        result = runner.invoke(app, ["create", "unused", "--list-templates"])
        assert result.exit_code == 0

    def test_list_templates_shorthand(self) -> None:
        # -l works without a project name (eager option fires before arg validation)
        result = runner.invoke(app, ["create", "-l"])
        assert result.exit_code == 0

    def test_list_templates_shows_all_template_values(self) -> None:
        result = runner.invoke(app, ["create", "unused", "--list-templates"])
        for t in Template:
            assert t.value in result.output

    def test_list_templates_shows_descriptions(self) -> None:
        result = runner.invoke(app, ["create", "unused", "--list-templates"])
        # Normalize whitespace to handle Rich line-wrapping
        normalized_output = " ".join(result.output.split())
        for t in Template:
            normalized_desc = " ".join(t.description.split())
            assert normalized_desc in normalized_output

    def test_list_templates_does_not_create_directory(self, tmp_path: Path) -> None:
        name = str(tmp_path / "should-not-exist")
        runner.invoke(app, ["create", name, "--list-templates"])
        assert not Path(name).exists()

    def test_invalid_template_exits_nonzero(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "create",
                str(tmp_path / "proj"),
                "--template",
                "bad-value",
                "--data",
                "synthetic",
                "--lightning",
            ],
        )
        assert result.exit_code != 0
        assert "bad-value" in result.output
        for t in Template:
            assert t.value in result.output

    def test_invalid_template_exit_code_is_2(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "create",
                str(tmp_path / "proj"),
                "--template",
                "notreal",
                "--data",
                "synthetic",
                "--lightning",
            ],
        )
        assert result.exit_code == 2

    def test_help_mentions_list_templates(self) -> None:
        result = runner.invoke(app, ["create", "--help"])
        # Strip ANSI codes before checking: on CI (GITHUB_ACTIONS=true) Typer's Rich
        # renderer forces terminal mode and splits hyphenated names with escape sequences,
        # so "--list-templates" would not appear as a contiguous plain-text substring.
        assert "--list-templates" in click.unstyle(result.output)
        assert result.exit_code == 0
