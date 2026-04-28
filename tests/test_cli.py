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

    def test_existing_empty_directory_succeeds(self, project_dir: Path) -> None:
        """Creating a project in an existing empty directory should succeed."""
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

        assert result.exit_code == 0
        assert (project_dir / "pyproject.toml").exists()

    def test_nonempty_directory_cancel_first_prompt(self, project_dir: Path) -> None:
        """Cancelling the first confirmation exits without deleting."""
        project_dir.mkdir(parents=True)
        (project_dir / "existing.txt").write_text("keep me")

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
            input="n\n",
        )

        assert result.exit_code == 1
        assert (project_dir / "existing.txt").exists()

    def test_nonempty_directory_cancel_second_prompt(self, project_dir: Path) -> None:
        """Accepting first but cancelling second confirmation exits without deleting."""
        project_dir.mkdir(parents=True)
        (project_dir / "existing.txt").write_text("keep me")

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
            input="y\nn\n",
        )

        assert result.exit_code == 1
        assert (project_dir / "existing.txt").exists()

    def test_nonempty_directory_confirm_deletes_and_creates(self, project_dir: Path) -> None:
        """Confirming both prompts deletes contents and creates project."""
        project_dir.mkdir(parents=True)
        (project_dir / "existing.txt").write_text("delete me")

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
            input="y\ny\n",
        )

        assert result.exit_code == 0
        assert not (project_dir / "existing.txt").exists()
        assert (project_dir / "pyproject.toml").exists()

    def test_create_dot_uses_current_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """'anypinn create .' uses the current directory."""
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(
            app,
            ["create", ".", "--template", "sir", "--data", "synthetic", "--lightning"],
        )

        assert result.exit_code == 0
        assert (tmp_path / "pyproject.toml").exists()

    def test_create_no_arg_uses_current_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """'anypinn create' with no project name uses the current directory."""
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(
            app,
            ["create", "--template", "sir", "--data", "synthetic", "--lightning"],
        )

        assert result.exit_code == 0
        assert (tmp_path / "pyproject.toml").exists()

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
        """Lightning project pyproject.toml must list tensorboard dependency."""
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
        assert '"tensorboard"' in pyproject

    def test_pyproject_excludes_lightning_deps(self, project_dir: Path) -> None:
        """Core-only project pyproject.toml must not list tensorboard dependency."""
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
        assert '"tensorboard"' not in pyproject

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

    def test_pyproject_torchdiffeq_is_transitive(self, project_dir: Path) -> None:
        """torchdiffeq is a transitive dep of anypinn, not listed in scaffold pyproject."""
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
