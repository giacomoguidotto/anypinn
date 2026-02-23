"""Collocation point sampling strategies for PINN training."""

from __future__ import annotations

import math
from typing import Protocol

import torch
from torch import Tensor

from anypinn.core.nn import Domain
from anypinn.core.types import CollocationStrategies


class CollocationSampler(Protocol):
    """Protocol for collocation point samplers.

    Implementations must return a tensor of shape ``(n, domain.ndim)`` with all
    points inside the domain bounds.
    """

    def sample(self, n: int, domain: Domain) -> Tensor: ...


class UniformSampler:
    """Cartesian grid sampler that distributes points evenly across the domain.

    For d-dimensional domains, places ``ceil(n^(1/d))`` points per axis then
    takes the first ``n`` points of the resulting grid.

    Args:
        seed: Optional seed (unused — grid is deterministic).
    """

    def __init__(self, seed: int | None = None) -> None:
        pass

    def sample(self, n: int, domain: Domain) -> Tensor:
        d = domain.ndim
        pts_per_dim = math.ceil(n ** (1.0 / d))

        linspaces = [torch.linspace(lo, hi, pts_per_dim) for lo, hi in domain.bounds]
        grids = torch.meshgrid(*linspaces, indexing="ij")
        flat = torch.stack([g.reshape(-1) for g in grids], dim=-1)
        return flat[:n]


class RandomSampler:
    """Uniform random sampler inside domain bounds.

    Args:
        seed: Optional seed for reproducible sampling.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._gen = torch.Generator()
        if seed is not None:
            self._gen.manual_seed(seed)

    def sample(self, n: int, domain: Domain) -> Tensor:
        d = domain.ndim
        u = torch.rand((n, d), generator=self._gen)
        for i, (lo, hi) in enumerate(domain.bounds):
            u[:, i] = u[:, i] * (hi - lo) + lo
        return u


class LatinHypercubeSampler:
    """Latin Hypercube sampler (pure-PyTorch, no SciPy dependency).

    Stratifies each dimension into ``n`` equal intervals and places one sample
    per interval, then shuffles columns independently.

    Args:
        seed: Optional seed for reproducible sampling.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._gen = torch.Generator()
        if seed is not None:
            self._gen.manual_seed(seed)

    def sample(self, n: int, domain: Domain) -> Tensor:
        d = domain.ndim
        result = torch.empty(n, d)

        for i, (lo, hi) in enumerate(domain.bounds):
            perm = torch.randperm(n, generator=self._gen)
            base = (perm.float() + torch.rand(n, generator=self._gen)) / n
            result[:, i] = base * (hi - lo) + lo

        return result


class LogUniform1DSampler:
    """Log-uniform sampler for 1-D domains (reproduces SIR collocation behavior).

    Samples uniformly in ``log1p`` space and maps back via ``expm1``, producing
    a distribution that is denser near the lower bound — useful for epidemic
    models where early dynamics are most informative.

    Args:
        seed: Optional seed for reproducible sampling.

    Raises:
        ValueError: If the domain is not 1-D or ``x0 <= -1``.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._gen = torch.Generator()
        if seed is not None:
            self._gen.manual_seed(seed)

    def sample(self, n: int, domain: Domain) -> Tensor:
        if domain.ndim != 1:
            raise ValueError(
                f"log_uniform_1d sampler supports only 1-D domains, got ndim={domain.ndim}."
            )
        x0, x1 = domain.x0, domain.x1
        if x0 <= -1.0:
            raise ValueError(f"log_uniform_1d requires x0 > -1 for log1p, got x0={x0}.")
        log_lo = torch.tensor(x0, dtype=torch.float32).log1p()
        log_hi = torch.tensor(x1, dtype=torch.float32).log1p()
        u = torch.rand((n, 1), generator=self._gen)
        return torch.expm1(u * (log_hi - log_lo) + log_lo)


class ResidualScorer(Protocol):
    """Protocol for scoring candidate collocation points by PDE residual magnitude."""

    def residual_score(self, x: Tensor) -> Tensor:
        """Return per-point non-negative residual score of shape ``(n,)``.

        Args:
            x: Candidate collocation points ``(n, d)``.

        Returns:
            Scores ``(n,)`` — higher means larger residual.
        """
        ...


class AdaptiveSampler:
    """Residual-weighted adaptive collocation sampler.

    Draws an oversample of candidate points, scores them using a
    ``ResidualScorer``, and retains the top-scoring subset. A configurable
    ``exploration_ratio`` ensures a fraction of purely random points to prevent
    mode collapse.

    Args:
        scorer: Callable returning per-point residual scores ``(n,)``.
        oversample_factor: Multiplier on ``n`` for candidate generation.
        exploration_ratio: Fraction of the budget reserved for random points.
        seed: Optional seed for reproducible sampling.
    """

    def __init__(
        self,
        scorer: ResidualScorer,
        oversample_factor: int = 4,
        exploration_ratio: float = 0.2,
        seed: int | None = None,
    ) -> None:
        if oversample_factor < 1:
            raise ValueError(f"oversample_factor must be >= 1, got {oversample_factor}.")
        if not (0.0 <= exploration_ratio <= 1.0):
            raise ValueError(f"exploration_ratio must be in [0, 1], got {exploration_ratio}.")
        self._scorer = scorer
        self._oversample = oversample_factor
        self._explore = exploration_ratio
        self._random = RandomSampler(seed=seed)

    def sample(self, n: int, domain: Domain) -> Tensor:
        n_explore = max(1, int(n * self._explore))
        n_exploit = n - n_explore

        explore_pts = self._random.sample(n_explore, domain)

        if n_exploit <= 0:
            return explore_pts

        n_candidates = n_exploit * self._oversample
        candidates = self._random.sample(n_candidates, domain)

        with torch.no_grad():
            scores = self._scorer.residual_score(candidates)

        _, top_idx = scores.topk(min(n_exploit, len(scores)))
        exploit_pts = candidates[top_idx]

        return torch.cat([explore_pts, exploit_pts], dim=0)


_SAMPLER_REGISTRY: dict[str, type] = {
    "uniform": UniformSampler,
    "random": RandomSampler,
    "latin_hypercube": LatinHypercubeSampler,
    "log_uniform_1d": LogUniform1DSampler,
}


def build_sampler(
    strategy: CollocationStrategies,
    seed: int | None = None,
    scorer: ResidualScorer | None = None,
) -> CollocationSampler:
    """Construct a collocation sampler from a strategy name.

    Args:
        strategy: One of the ``CollocationStrategies`` literals.
        seed: Optional seed for reproducible sampling.
        scorer: Required when ``strategy="adaptive"``.

    Returns:
        A sampler instance satisfying the ``CollocationSampler`` protocol.

    Raises:
        ValueError: If ``strategy="adaptive"`` but no scorer is provided.
    """
    if strategy == "adaptive":
        if scorer is None:
            raise ValueError(
                "AdaptiveSampler requires a ResidualScorer. "
                "Pass a scorer via PINNDataModule or use a different strategy."
            )
        return AdaptiveSampler(scorer=scorer, seed=seed)

    cls = _SAMPLER_REGISTRY.get(strategy)
    if cls is None:
        raise ValueError(
            f"Unknown collocation strategy '{strategy}'. "
            f"Choose from: {', '.join(_SAMPLER_REGISTRY)} or 'adaptive'."
        )
    return cls(seed=seed)
