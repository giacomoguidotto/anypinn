"""Unit tests for the project renderer."""

from __future__ import annotations

from pathlib import Path

import pytest

from anypinn.cli._renderer import render_project
from anypinn.cli._types import DataSource, Template


class TestRenderProject:
    def test_creates_directory_structure(self, tmp_path: Path) -> None:
        project = tmp_path / "my-project"
        created = render_project(project, Template.SIR, DataSource.SYNTHETIC, True)

        assert project.is_dir()
        assert (project / "data").is_dir()
        assert "ode.py" in created
        assert "config.py" in created
        assert "train.py" in created
        assert "data/" in created

    def test_raises_on_existing_directory(self, tmp_path: Path) -> None:
        project = tmp_path / "existing"
        project.mkdir()

        with pytest.raises(FileExistsError):
            render_project(project, Template.BLANK, DataSource.SYNTHETIC, True)

    @pytest.mark.parametrize("template", list(Template))
    def test_all_templates_produce_valid_python(self, tmp_path: Path, template: Template) -> None:
        project = tmp_path / f"proj-{template.value}"
        render_project(project, template, DataSource.SYNTHETIC, True)

        for py_file in project.glob("*.py"):
            source = py_file.read_text()
            compile(source, str(py_file), "exec")

    def test_sir_ode_contains_sir_keys(self, tmp_path: Path) -> None:
        project = tmp_path / "sir-test"
        render_project(project, Template.SIR, DataSource.SYNTHETIC, True)

        ode_content = (project / "ode.py").read_text()
        assert 'S_KEY = "S"' in ode_content
        assert 'I_KEY = "I"' in ode_content
        assert 'BETA_KEY = "beta"' in ode_content

    def test_seir_ode_contains_seir_keys(self, tmp_path: Path) -> None:
        project = tmp_path / "seir-test"
        render_project(project, Template.SEIR, DataSource.SYNTHETIC, True)

        ode_content = (project / "ode.py").read_text()
        assert 'E_KEY = "E"' in ode_content
        assert 'SIGMA_KEY = "sigma"' in ode_content

    def test_damped_oscillator_keys(self, tmp_path: Path) -> None:
        project = tmp_path / "osc-test"
        render_project(project, Template.DAMPED_OSCILLATOR, DataSource.SYNTHETIC, True)

        ode_content = (project / "ode.py").read_text()
        assert 'ZETA_KEY = "zeta"' in ode_content
        assert 'OMEGA_KEY = "omega0"' in ode_content

    def test_lotka_volterra_keys(self, tmp_path: Path) -> None:
        project = tmp_path / "lv-test"
        render_project(project, Template.LOTKA_VOLTERRA, DataSource.SYNTHETIC, True)

        ode_content = (project / "ode.py").read_text()
        assert 'ALPHA_KEY = "alpha"' in ode_content
        assert "fourier_encode" in ode_content

    def test_custom_contains_todos(self, tmp_path: Path) -> None:
        project = tmp_path / "custom-test"
        render_project(project, Template.CUSTOM, DataSource.SYNTHETIC, True)

        ode_content = (project / "ode.py").read_text()
        assert "TODO" in ode_content

    def test_blank_is_minimal(self, tmp_path: Path) -> None:
        project = tmp_path / "blank-test"
        render_project(project, Template.BLANK, DataSource.SYNTHETIC, True)

        ode_content = (project / "ode.py").read_text()
        assert "NotImplementedError" in ode_content
