"""Tests for anypinn.core.samplers — collocation sampling strategies."""

import pytest
import torch

from anypinn.core.nn import Domain
from anypinn.core.samplers import (
    AdaptiveSampler,
    LatinHypercubeSampler,
    LogUniform1DSampler,
    RandomSampler,
    UniformSampler,
    build_sampler,
)


def _1d_domain() -> Domain:
    return Domain(bounds=[(0.0, 10.0)], dx=[0.1])


def _2d_domain() -> Domain:
    return Domain(bounds=[(0.0, 1.0), (2.0, 5.0)])


def _3d_domain() -> Domain:
    return Domain(bounds=[(-1.0, 1.0), (0.0, 3.14), (10.0, 20.0)])


class _MockScorer:
    """Trivial scorer returning random scores for testing AdaptiveSampler."""

    def residual_score(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rand(x.shape[0])


class TestUniformSampler:
    def test_1d_shape(self):
        s = UniformSampler()
        pts = s.sample(100, _1d_domain())
        assert pts.shape == (100, 1)

    def test_2d_shape(self):
        s = UniformSampler()
        pts = s.sample(50, _2d_domain())
        assert pts.shape[1] == 2
        assert pts.shape[0] >= 50

    def test_3d_shape(self):
        s = UniformSampler()
        pts = s.sample(27, _3d_domain())
        assert pts.shape[1] == 3

    def test_points_within_bounds(self):
        domain = _2d_domain()
        pts = UniformSampler().sample(100, domain)
        for i, (lo, hi) in enumerate(domain.bounds):
            assert pts[:, i].min().item() >= lo - 1e-6
            assert pts[:, i].max().item() <= hi + 1e-6

    def test_grid_is_deterministic_regardless_of_seed(self):
        """Grid output is always the same — seed is accepted but ignored."""
        pts_no_seed = UniformSampler().sample(10, _1d_domain())
        pts_seed_42 = UniformSampler(seed=42).sample(10, _1d_domain())
        pts_seed_99 = UniformSampler(seed=99).sample(10, _1d_domain())
        assert torch.equal(pts_no_seed, pts_seed_42)
        assert torch.equal(pts_no_seed, pts_seed_99)


class TestRandomSampler:
    def test_1d_shape(self):
        pts = RandomSampler(seed=0).sample(100, _1d_domain())
        assert pts.shape == (100, 1)

    def test_2d_shape(self):
        pts = RandomSampler(seed=0).sample(100, _2d_domain())
        assert pts.shape == (100, 2)

    def test_3d_shape(self):
        pts = RandomSampler(seed=0).sample(50, _3d_domain())
        assert pts.shape == (50, 3)

    def test_points_within_bounds(self):
        domain = _2d_domain()
        pts = RandomSampler(seed=0).sample(1000, domain)
        for i, (lo, hi) in enumerate(domain.bounds):
            assert pts[:, i].min().item() >= lo
            assert pts[:, i].max().item() <= hi

    def test_deterministic_with_seed(self):
        a = RandomSampler(seed=42).sample(100, _1d_domain())
        b = RandomSampler(seed=42).sample(100, _1d_domain())
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        a = RandomSampler(seed=1).sample(100, _1d_domain())
        b = RandomSampler(seed=2).sample(100, _1d_domain())
        assert not torch.equal(a, b)


class TestLatinHypercubeSampler:
    def test_1d_shape(self):
        pts = LatinHypercubeSampler(seed=0).sample(100, _1d_domain())
        assert pts.shape == (100, 1)

    def test_2d_shape(self):
        pts = LatinHypercubeSampler(seed=0).sample(100, _2d_domain())
        assert pts.shape == (100, 2)

    def test_points_within_bounds(self):
        domain = _2d_domain()
        pts = LatinHypercubeSampler(seed=0).sample(200, domain)
        for i, (lo, hi) in enumerate(domain.bounds):
            assert pts[:, i].min().item() >= lo - 1e-6
            assert pts[:, i].max().item() <= hi + 1e-6

    def test_stratification_property(self):
        """Each interval [j/n, (j+1)/n) should contain exactly one sample per dim."""
        n = 50
        domain = Domain(bounds=[(0.0, 1.0)])
        pts = LatinHypercubeSampler(seed=0).sample(n, domain)
        bins = (pts[:, 0] * n).long().clamp(0, n - 1)
        assert len(bins.unique()) == n

    def test_deterministic_with_seed(self):
        a = LatinHypercubeSampler(seed=7).sample(50, _2d_domain())
        b = LatinHypercubeSampler(seed=7).sample(50, _2d_domain())
        assert torch.equal(a, b)


class TestLogUniform1DSampler:
    def test_shape(self):
        pts = LogUniform1DSampler(seed=0).sample(200, _1d_domain())
        assert pts.shape == (200, 1)

    def test_points_within_bounds(self):
        domain = _1d_domain()
        pts = LogUniform1DSampler(seed=0).sample(1000, domain)
        assert pts.min().item() >= domain.x0 - 1e-5
        assert pts.max().item() <= domain.x1 + 1e-5

    def test_rejects_multidim_domain(self):
        with pytest.raises(ValueError, match="1-D domains"):
            LogUniform1DSampler().sample(10, _2d_domain())

    def test_rejects_invalid_x0(self):
        domain = Domain(bounds=[(-2.0, 5.0)])
        with pytest.raises(ValueError, match="x0 > -1"):
            LogUniform1DSampler().sample(10, domain)

    def test_deterministic_with_seed(self):
        a = LogUniform1DSampler(seed=42).sample(100, _1d_domain())
        b = LogUniform1DSampler(seed=42).sample(100, _1d_domain())
        assert torch.equal(a, b)

    def test_denser_near_lower_bound(self):
        """Log-uniform should place more points near the start of the domain."""
        domain = Domain(bounds=[(0.0, 100.0)])
        pts = LogUniform1DSampler(seed=0).sample(10000, domain)
        below_midpoint = (pts < 50.0).sum().item()
        assert below_midpoint > 5000


class TestAdaptiveSampler:
    def test_shape(self):
        scorer = _MockScorer()
        s = AdaptiveSampler(scorer=scorer, seed=0)
        pts = s.sample(100, _1d_domain())
        assert pts.shape == (100, 1)

    def test_2d_shape(self):
        scorer = _MockScorer()
        s = AdaptiveSampler(scorer=scorer, seed=0)
        pts = s.sample(100, _2d_domain())
        assert pts.shape == (100, 2)

    def test_exploration_ratio_respected(self):
        """With exploration_ratio=1.0, all points should be random."""
        scorer = _MockScorer()
        s = AdaptiveSampler(scorer=scorer, exploration_ratio=1.0, seed=0)
        pts = s.sample(50, _1d_domain())
        assert pts.shape == (50, 1)

    def test_invalid_oversample_factor_raises(self):
        with pytest.raises(ValueError, match="oversample_factor"):
            AdaptiveSampler(scorer=_MockScorer(), oversample_factor=0)

    def test_invalid_exploration_ratio_raises(self):
        with pytest.raises(ValueError, match="exploration_ratio"):
            AdaptiveSampler(scorer=_MockScorer(), exploration_ratio=1.5)


class TestBuildSampler:
    def test_random(self):
        s = build_sampler("random", seed=0)
        pts = s.sample(10, _1d_domain())
        assert pts.shape == (10, 1)

    def test_uniform(self):
        s = build_sampler("uniform")
        pts = s.sample(10, _1d_domain())
        assert pts.shape == (10, 1)

    def test_latin_hypercube(self):
        s = build_sampler("latin_hypercube", seed=0)
        pts = s.sample(10, _1d_domain())
        assert pts.shape == (10, 1)

    def test_log_uniform_1d(self):
        s = build_sampler("log_uniform_1d", seed=0)
        pts = s.sample(10, _1d_domain())
        assert pts.shape == (10, 1)

    def test_adaptive_requires_scorer(self):
        with pytest.raises(ValueError, match="ResidualScorer"):
            build_sampler("adaptive")

    def test_adaptive_with_scorer(self):
        s = build_sampler("adaptive", scorer=_MockScorer(), seed=0)
        pts = s.sample(50, _1d_domain())
        assert pts.shape == (50, 1)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown collocation strategy"):
            build_sampler("nonexistent")  # type: ignore[arg-type]
