"""Enums for CLI options."""

from enum import Enum


class Template(str, Enum):
    """Available project templates."""

    SIR = "sir"
    SEIR = "seir"
    DAMPED_OSCILLATOR = "damped-oscillator"
    LOTKA_VOLTERRA = "lotka-volterra"
    VAN_DER_POL = "van-der-pol"
    LORENZ = "lorenz"
    POISSON_2D = "poisson-2d"
    HEAT_1D = "heat-1d"
    BURGERS_1D = "burgers-1d"
    WAVE_1D = "wave-1d"
    CUSTOM = "custom"
    BLANK = "blank"

    @property
    def label(self) -> str:
        labels: dict[Template, str] = {
            Template.SIR: "SIR Epidemic Model",
            Template.SEIR: "SEIR Epidemic Model",
            Template.DAMPED_OSCILLATOR: "Damped Oscillator",
            Template.LOTKA_VOLTERRA: "Lotka-Volterra",
            Template.VAN_DER_POL: "Van der Pol Oscillator",
            Template.LORENZ: "Lorenz System",
            Template.POISSON_2D: "Poisson 2D",
            Template.HEAT_1D: "Heat Equation 1D",
            Template.BURGERS_1D: "Burgers Equation 1D",
            Template.WAVE_1D: "Wave Equation 1D",
            Template.CUSTOM: "Custom ODE",
            Template.BLANK: "Blank project",
        }
        return labels[self]

    @property
    def description(self) -> str:
        descriptions: dict[Template, str] = {
            Template.SIR: "Classic S→I→R compartmental model. Learns transmission rate β.",
            Template.SEIR: "Extended epidemic model with exposed compartment E. Learns β.",
            Template.DAMPED_OSCILLATOR: "Harmonic oscillator with damping. Learns damping ratio ζ.",  # noqa: E501
            Template.LOTKA_VOLTERRA: "Predator-prey dynamics with Fourier encoding. Learns predation rate β.",  # noqa: E501
            Template.VAN_DER_POL: "Second-order nonlinear oscillator. Learns nonlinearity parameter μ.",  # noqa: E501
            Template.LORENZ: "Chaotic 3-field ODE. Learns sigma, rho, beta with Huber criterion.",
            Template.POISSON_2D: "2D elliptic PDE forward problem. Demonstrates PDEResidualConstraint + DirichletBC.",  # noqa: E501
            Template.HEAT_1D: "1D parabolic PDE inverse problem. Recovers thermal diffusivity from sparse measurements.",  # noqa: E501
            Template.BURGERS_1D: "1D nonlinear PDE with shock formation. Recovers viscosity with adaptive collocation.",  # noqa: E501
            Template.WAVE_1D: "1D hyperbolic PDE inverse problem. Recovers wave speed from sparse measurements.",  # noqa: E501
            Template.CUSTOM: "Minimal skeleton for a user-defined ODE. All factories are stubs.",
            Template.BLANK: "Empty project structure with no ODE—start from scratch.",
        }
        return descriptions[self]


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
