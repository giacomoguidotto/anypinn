"""Enums for CLI options."""

from enum import Enum


class Template(str, Enum):
    """Available project templates."""

    SIR = "sir"
    SEIR = "seir"
    DAMPED_OSCILLATOR = "damped-oscillator"
    LOTKA_VOLTERRA = "lotka-volterra"
    CUSTOM = "custom"
    BLANK = "blank"

    @property
    def label(self) -> str:
        labels: dict[Template, str] = {
            Template.SIR: "SIR Epidemic Model",
            Template.SEIR: "SEIR Epidemic Model",
            Template.DAMPED_OSCILLATOR: "Damped Oscillator",
            Template.LOTKA_VOLTERRA: "Lotka-Volterra",
            Template.CUSTOM: "Custom ODE",
            Template.BLANK: "Blank project",
        }
        return labels[self]


class DataSource(str, Enum):
    """Training data source."""

    SYNTHETIC = "synthetic"
    CSV = "csv"

    @property
    def label(self) -> str:
        labels: dict[DataSource, str] = {
            DataSource.SYNTHETIC: "Generate synthetic data",
            DataSource.CSV: "Load from CSV",
        }
        return labels[self]
