"""AnyPINN: Physics-Informed Neural Networks library."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("anypinn")
except PackageNotFoundError:
    __version__ = "0.0.0"
