"""Tests for anypinn.core.config — Config dataclasses."""

from pathlib import Path

import torch

from anypinn.core.config import (
    EarlyStoppingConfig,
    GenerationConfig,
    IngestionConfig,
    MLPConfig,
    PINNHyperparameters,
    ScalarConfig,
    SchedulerConfig,
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


class TestSchedulerConfig:
    def test_instantiation(self):
        c = SchedulerConfig(mode="min", factor=0.5, patience=10, threshold=1e-4, min_lr=1e-6)
        assert c.mode == "min"
        assert c.factor == 0.5


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
