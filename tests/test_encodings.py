"""Tests for anypinn.lib.encodings and Field integration."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from anypinn.core.config import MLPConfig
from anypinn.core.nn import Field
from anypinn.lib.encodings import FourierEncoding, RandomFourierFeatures


class TestFourierEncoding:
    def test_output_shape_include_input(self):
        enc = FourierEncoding(num_frequencies=4, include_input=True)
        x = torch.randn(10, 3)
        out = enc(x)
        # (n, d*(1 + 2K)) = (10, 3*(1+8)) = (10, 27)
        assert out.shape == (10, 27)

    def test_output_shape_exclude_input(self):
        enc = FourierEncoding(num_frequencies=4, include_input=False)
        x = torch.randn(10, 3)
        out = enc(x)
        # (n, d*2K) = (10, 3*8) = (10, 24)
        assert out.shape == (10, 24)

    def test_out_dim_helper_include_input(self):
        enc = FourierEncoding(num_frequencies=4, include_input=True)
        assert enc.out_dim(3) == 27
        x = torch.randn(5, 3)
        assert enc(x).shape[-1] == enc.out_dim(3)

    def test_out_dim_helper_exclude_input(self):
        enc = FourierEncoding(num_frequencies=4, include_input=False)
        assert enc.out_dim(3) == 24
        x = torch.randn(5, 3)
        assert enc(x).shape[-1] == enc.out_dim(3)

    def test_1d_input(self):
        enc = FourierEncoding(num_frequencies=6, include_input=True)
        x = torch.randn(100, 1)
        out = enc(x)
        # (100, 1*(1+12)) = (100, 13)
        assert out.shape == (100, 13)
        assert enc.out_dim(1) == 13

    def test_multidim_input(self):
        enc = FourierEncoding(num_frequencies=3, include_input=True)
        x = torch.randn(50, 2)
        out = enc(x)
        # (50, 2*(1+6)) = (50, 14)
        assert out.shape == (50, 14)

    def test_invalid_num_frequencies_raises(self):
        with pytest.raises(ValueError, match="num_frequencies must be >= 1"):
            FourierEncoding(num_frequencies=0)

    def test_negative_num_frequencies_raises(self):
        with pytest.raises(ValueError, match="num_frequencies must be >= 1"):
            FourierEncoding(num_frequencies=-1)


class TestRandomFourierFeatures:
    def test_output_shape(self):
        rff = RandomFourierFeatures(in_dim=2, num_features=64)
        x = torch.randn(100, 2)
        out = rff(x)
        assert out.shape == (100, 128)

    def test_out_dim_property(self):
        rff = RandomFourierFeatures(in_dim=3, num_features=32)
        assert rff.out_dim == 64

    def test_deterministic_with_seed(self):
        x = torch.randn(20, 2)
        rff1 = RandomFourierFeatures(in_dim=2, num_features=16, seed=42)
        rff2 = RandomFourierFeatures(in_dim=2, num_features=16, seed=42)
        assert torch.allclose(rff1(x), rff2(x))

    def test_different_seeds_differ(self):
        x = torch.randn(20, 2)
        rff1 = RandomFourierFeatures(in_dim=2, num_features=64, seed=1)
        rff2 = RandomFourierFeatures(in_dim=2, num_features=64, seed=2)
        assert not torch.allclose(rff1(x), rff2(x))

    def test_buffer_moves_with_module(self):
        rff = RandomFourierFeatures(in_dim=2, num_features=16, seed=0)
        rff_cpu = rff.to("cpu")
        assert rff_cpu.B.device.type == "cpu"
        # B is still a registered buffer
        assert "B" in dict(rff_cpu.named_buffers())

    def test_invalid_in_dim_raises(self):
        with pytest.raises(ValueError, match="in_dim must be >= 1"):
            RandomFourierFeatures(in_dim=0)

    def test_invalid_num_features_raises(self):
        with pytest.raises(ValueError, match="num_features must be >= 1"):
            RandomFourierFeatures(in_dim=1, num_features=0)

    def test_invalid_scale_raises(self):
        with pytest.raises(ValueError, match="scale must be > 0"):
            RandomFourierFeatures(in_dim=1, num_features=8, scale=0.0)

    def test_negative_scale_raises(self):
        with pytest.raises(ValueError, match="scale must be > 0"):
            RandomFourierFeatures(in_dim=1, num_features=8, scale=-1.0)


class TestFieldWithEncoding:
    def test_field_with_fourier_encoding(self):
        enc = FourierEncoding(num_frequencies=4, include_input=True)
        # in_dim for the MLP must match enc.out_dim(1) = 9
        config = MLPConfig(
            in_dim=enc.out_dim(1), out_dim=3, hidden_layers=[16], activation="tanh", encode=enc
        )
        field = Field(config)
        x = torch.randn(20, 1)
        out = field(x)
        assert out.shape == (20, 3)

    def test_field_with_rff_encoding(self):
        rff = RandomFourierFeatures(in_dim=1, num_features=32, seed=0)
        # MLP in_dim = 2*32 = 64
        config = MLPConfig(
            in_dim=rff.out_dim, out_dim=2, hidden_layers=[16], activation="tanh", encode=rff
        )
        field = Field(config)
        x = torch.randn(15, 1)
        out = field(x)
        assert out.shape == (15, 2)

    def test_encoder_registered_as_submodule_fourier(self):
        enc = FourierEncoding(num_frequencies=3)
        config = MLPConfig(
            in_dim=enc.out_dim(1), out_dim=1, hidden_layers=[8], activation="tanh", encode=enc
        )
        field = Field(config)
        assert isinstance(field.encoder, nn.Module)
        submodule_names = [name for name, _ in field.named_modules()]
        assert "encoder" in submodule_names

    def test_encoder_registered_as_submodule_rff(self):
        rff = RandomFourierFeatures(in_dim=1, num_features=8, seed=0)
        config = MLPConfig(
            in_dim=rff.out_dim, out_dim=1, hidden_layers=[8], activation="tanh", encode=rff
        )
        field = Field(config)
        assert isinstance(field.encoder, nn.Module)
        assert "encoder" in dict(field.named_modules())

    def test_plain_callable_still_works(self):
        # Backward-compat: plain lambda encode passes through unchanged
        double_fn = lambda x: x * 2  # noqa: E731
        config = MLPConfig(
            in_dim=1, out_dim=1, hidden_layers=[8], activation="tanh", encode=double_fn
        )
        field = Field(config)
        assert field.encoder is None
        assert field._encode_fn is double_fn
        x = torch.randn(10, 1)
        out = field(x)
        assert out.shape == (10, 1)

    def test_no_encoding_works(self):
        config = MLPConfig(in_dim=1, out_dim=2, hidden_layers=[8], activation="tanh")
        field = Field(config)
        assert field.encoder is None
        assert field._encode_fn is None
        x = torch.randn(10, 1)
        out = field(x)
        assert out.shape == (10, 2)
