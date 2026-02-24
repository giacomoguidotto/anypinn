"""Built-in input encodings for spatial/periodic signals."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn as nn


class FourierEncoding(nn.Module):
    """Sinusoidal positional encoding for periodic or high-frequency signals.

    For input :math:`\\mathbf{x} \\in \\mathbb{R}^{n \\times d}` and
    ``num_frequencies`` :math:`K`, the encoding is:

    .. math::

        \\gamma(\\mathbf{x}) = [\\mathbf{x},\\,
            \\sin(\\mathbf{x}),\\, \\cos(\\mathbf{x}),\\,
            \\sin(2\\mathbf{x}),\\, \\cos(2\\mathbf{x}),\\,
            \\ldots,\\,
            \\sin(K\\mathbf{x}),\\, \\cos(K\\mathbf{x})]

    producing shape :math:`(n,\\, d\\,(1 + 2K))` when ``include_input=True``,
    or :math:`(n,\\, 2dK)` when ``include_input=False``.

    Args:
        num_frequencies: Number of frequency bands :math:`K \\geq 1`.
        include_input:   Prepend original coordinates to the encoded output.
    """

    def __init__(self, num_frequencies: int = 6, include_input: bool = True) -> None:
        if num_frequencies < 1:
            raise ValueError(f"num_frequencies must be >= 1, got {num_frequencies}.")
        super().__init__()
        self.num_frequencies = num_frequencies
        self.include_input = include_input

    def out_dim(self, in_dim: int) -> int:
        """Compute output dimension given input dimension."""
        factor = 1 + 2 * self.num_frequencies if self.include_input else 2 * self.num_frequencies
        return in_dim * factor

    def forward(self, x: Tensor) -> Tensor:
        parts = [x] if self.include_input else []
        for k in range(1, self.num_frequencies + 1):
            parts.append(torch.sin(k * x))
            parts.append(torch.cos(k * x))
        return torch.cat(parts, dim=-1)


class RandomFourierFeatures(nn.Module):
    """Random Fourier Features (Rahimi & Recht, 2007) for RBF kernel approximation.

    Draws a fixed random matrix :math:`\\mathbf{B} \\sim \\mathcal{N}(0, \\sigma^2)`
    of shape :math:`(d_{\\text{in}},\\, m)` and maps
    :math:`\\mathbf{x} \\in \\mathbb{R}^{n \\times d_{\\text{in}}}` to:

    .. math::

        \\phi(\\mathbf{x}) = \\frac{1}{\\sqrt{m}}
            [\\cos(\\mathbf{x}\\mathbf{B}),\\; \\sin(\\mathbf{x}\\mathbf{B})]
            \\in \\mathbb{R}^{n \\times 2m}

    :math:`\\mathbf{B}` is registered as a buffer and moves with the module across devices.

    Args:
        in_dim:       Spatial dimension :math:`d_{\\text{in}}` of the input.
        num_features: Number of random features :math:`m`
                      (output dimension :math:`= 2m`).
        scale:        Standard deviation :math:`\\sigma` of the frequency distribution.
                      Higher values capture higher-frequency variation. Default: 1.0.
        seed:         Optional seed for reproducible frequency sampling.
    """

    def __init__(
        self,
        in_dim: int,
        num_features: int = 256,
        scale: float = 1.0,
        seed: int | None = None,
    ) -> None:
        if in_dim < 1:
            raise ValueError(f"in_dim must be >= 1, got {in_dim}.")
        if num_features < 1:
            raise ValueError(f"num_features must be >= 1, got {num_features}.")
        if scale <= 0.0:
            raise ValueError(f"scale must be > 0, got {scale}.")
        super().__init__()
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        B = torch.randn(in_dim, num_features, generator=gen) * scale
        self.register_buffer("B", B)
        self.num_features = num_features

    @property
    def out_dim(self) -> int:
        """Output dimension (always 2 * num_features)."""
        return 2 * self.num_features

    def forward(self, x: Tensor) -> Tensor:
        proj = x @ self.B  # type: ignore[operator]  # (n, num_features)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1) / (self.num_features**0.5)
