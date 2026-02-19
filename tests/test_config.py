"""Tests for anypinn.core.config — Config dataclasses."""

from pathlib import Path

import torch

from anypinn.core.config import (
    AdamConfig,
    CosineAnnealingConfig,
    EarlyStoppingConfig,
    GenerationConfig,
    IngestionConfig,
    LBFGSConfig,
    MLPConfig,
    PINNHyperparameters,
    ReduceLROnPlateauConfig,
    ScalarConfig,
    SMMAStoppingConfig,
    TrainingDataConfig,
)


class TestMLPConfig:
    def test_instantiation(self):
        c = MLPConfig(in_dim=1, out_dim=3, hidden_layers=[32, 32], activation="tanh")
        assert c.in_dim == 1
        assert c.out_dim == 3
        assert c.hidden_layers == [32, 32]
        assert c.output_activation is None
        assert c.encode is None

    def test_with_output_activation(self):
        c = MLPConfig(
            in_dim=1, out_dim=1, hidden_layers=[8], activation="relu", output_activation="sigmoid"
        )
        assert c.output_activation == "sigmoid"


class TestScalarConfig:
    def test_instantiation(self):
        c = ScalarConfig(init_value=0.1)
        assert c.init_value == 0.1


class TestAdamConfig:
    def test_defaults(self):
        c = AdamConfig()
        assert c.lr == 1e-3
        assert c.betas == (0.9, 0.999)
        assert c.weight_decay == 0.0

    def test_custom(self):
        c = AdamConfig(lr=1e-4, betas=(0.8, 0.99), weight_decay=1e-5)
        assert c.lr == 1e-4


class TestLBFGSConfig:
    def test_defaults(self):
        c = LBFGSConfig()
        assert c.lr == 1.0
        assert c.max_iter == 20
        assert c.line_search_fn == "strong_wolfe"

    def test_custom(self):
        c = LBFGSConfig(lr=0.5, max_iter=50, history_size=200)
        assert c.max_iter == 50


class TestReduceLROnPlateauConfig:
    def test_instantiation(self):
        c = ReduceLROnPlateauConfig(
            mode="min", factor=0.5, patience=10, threshold=1e-4, min_lr=1e-6
        )
        assert c.mode == "min"
        assert c.factor == 0.5


class TestCosineAnnealingConfig:
    def test_instantiation(self):
        c = CosineAnnealingConfig(T_max=100)
        assert c.T_max == 100
        assert c.eta_min == 0.0

    def test_custom(self):
        c = CosineAnnealingConfig(T_max=200, eta_min=1e-6)
        assert c.eta_min == 1e-6


class TestEarlyStoppingConfig:
    def test_instantiation(self):
        c = EarlyStoppingConfig(patience=20, mode="min")
        assert c.patience == 20


class TestSMMAStoppingConfig:
    def test_instantiation(self):
        c = SMMAStoppingConfig(window=50, threshold=0.01, lookback=100)
        assert c.window == 50


class TestTrainingDataConfig:
    def test_instantiation(self):
        c = TrainingDataConfig(batch_size=64, data_ratio=0.5, collocations=200)
        assert c.batch_size == 64


class TestIngestionConfig:
    def test_instantiation(self):
        c = IngestionConfig(
            batch_size=32,
            data_ratio=0.5,
            collocations=100,
            df_path=Path("data.csv"),
            y_columns=["I"],
        )
        assert c.df_path == Path("data.csv")
        assert c.x_column is None
        assert c.x_transform is None


class TestGenerationConfig:
    def test_instantiation(self):
        c = GenerationConfig(
            batch_size=32,
            data_ratio=0.5,
            collocations=100,
            x=torch.linspace(0, 10, 50),
            noise_level=0.0,
            args_to_train={},
        )
        assert c.noise_level == 0.0
        assert c.x.shape == (50,)


class TestPINNHyperparameters:
    def test_instantiation(self):
        td = GenerationConfig(
            batch_size=32,
            data_ratio=0.5,
            collocations=100,
            x=torch.linspace(0, 10, 50),
            noise_level=0.0,
            args_to_train={},
        )
        hp = PINNHyperparameters(
            lr=1e-3,
            training_data=td,
            fields_config=MLPConfig(in_dim=1, out_dim=1, hidden_layers=[8], activation="tanh"),
            params_config=ScalarConfig(init_value=0.5),
        )
        assert hp.lr == 1e-3
        assert hp.scheduler is None
        assert hp.early_stopping is None
        assert hp.smma_stopping is None
