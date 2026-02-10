"""Unit tests for interactive prompt functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from anypinn.cli._prompts import prompt_data_source, prompt_lightning, prompt_template
from anypinn.cli._types import DataSource, Template


class TestPromptTemplate:
    @patch("anypinn.cli._prompts.TerminalMenu")
    def test_returns_selected_template(self, mock_menu_cls: MagicMock) -> None:
        mock_menu_cls.return_value.show.return_value = 0  # SIR

        result = prompt_template()
        assert result is Template.SIR

    @patch("anypinn.cli._prompts.TerminalMenu")
    def test_returns_last_template(self, mock_menu_cls: MagicMock) -> None:
        mock_menu_cls.return_value.show.return_value = 5  # BLANK

        result = prompt_template()
        assert result is Template.BLANK

    @patch("anypinn.cli._prompts.TerminalMenu")
    def test_exit_on_none(self, mock_menu_cls: MagicMock) -> None:
        mock_menu_cls.return_value.show.return_value = None

        with pytest.raises(SystemExit):
            prompt_template()


class TestPromptDataSource:
    @patch("anypinn.cli._prompts.TerminalMenu")
    def test_returns_synthetic(self, mock_menu_cls: MagicMock) -> None:
        mock_menu_cls.return_value.show.return_value = 0

        result = prompt_data_source()
        assert result is DataSource.SYNTHETIC

    @patch("anypinn.cli._prompts.TerminalMenu")
    def test_returns_csv(self, mock_menu_cls: MagicMock) -> None:
        mock_menu_cls.return_value.show.return_value = 1

        result = prompt_data_source()
        assert result is DataSource.CSV


class TestPromptLightning:
    @patch("builtins.input", return_value="")
    def test_default_yes(self, mock_input: MagicMock) -> None:
        result = prompt_lightning()
        assert result is True
        mock_input.assert_called_once()

    @patch("builtins.input", return_value="n")
    def test_explicit_no(self, mock_input: MagicMock) -> None:
        result = prompt_lightning()
        assert result is False
        mock_input.assert_called_once()

    @patch("builtins.input", return_value="y")
    def test_explicit_yes(self, mock_input: MagicMock) -> None:
        result = prompt_lightning()
        assert result is True
        mock_input.assert_called_once()
