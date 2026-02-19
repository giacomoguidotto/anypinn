"""Tests for anypinn.lightning.module — PINNModule."""

import torch
from torch import Tensor
import torch.nn as nn

from anypinn.core.config import (
    AdamConfig,
    CosineAnnealingConfig,
    GenerationConfig,
    LBFGSConfig,
    MLPConfig,
    PINNHyperparameters,
    ReduceLROnPlateauConfig,
    ScalarConfig,
)
from anypinn.core.context import InferredContext
from anypinn.core.nn import Field, Parameter
from anypinn.core.problem import Constraint, Problem
from anypinn.core.types import LogFn, TrainingBatch
from anypinn.lightning.module import PINNModule


class FixedConstraint(Constraint):
    def loss(self, batch: TrainingBatch, criterion: nn.Module, log: LogFn | None = None) -> Tensor:
        return torch.tensor(1.0)


def _make_hp(
    scheduler: ReduceLROnPlateauConfig | CosineAnnealingConfig | None = None,
    optimizer: AdamConfig | LBFGSConfig | None = None,
) -> PINNHyperparameters:
    return PINNHyperparameters(
        lr=1e-3,
        training_data=GenerationConfig(
            batch_size=16,
            data_ratio=0.5,
            collocations=50,
            x=torch.linspace(0, 10, 50),
            noise_level=0.0,
            args_to_train={},
        ),
        fields_config=MLPConfig(in_dim=1, out_dim=1, hidden_layers=[8], activation="tanh"),
        params_config=ScalarConfig(init_value=0.5),
        optimizer=optimizer,
        scheduler=scheduler,
    )


def _make_problem() -> Problem:
    fields = {"u": Field(MLPConfig(in_dim=1, out_dim=1, hidden_layers=[8], activation="tanh"))}
    params = {"alpha": Parameter(ScalarConfig(init_value=0.5))}
    return Problem(
        constraints=[FixedConstraint()],
        criterion=nn.MSELoss(),
        fields=fields,
        params=params,
    )


class TestPINNModule:
    def test_configure_optimizers_no_scheduler(self):
        hp = _make_hp()
        module = PINNModule(_make_problem(), hp)
        result = module.configure_optimizers()
        assert isinstance(result, torch.optim.Adam)

    def test_configure_optimizers_with_scheduler(self):
        sch = ReduceLROnPlateauConfig(
            mode="min", factor=0.5, patience=10, threshold=1e-4, min_lr=1e-6
        )
        hp = _make_hp(scheduler=sch)
        module = PINNModule(_make_problem(), hp)
        result = module.configure_optimizers()
        assert isinstance(result, dict)
        assert "optimizer" in result
        assert "lr_scheduler" in result

    def test_configure_optimizers_with_adam_config(self):
        hp = _make_hp(optimizer=AdamConfig(lr=1e-4, weight_decay=1e-5))
        module = PINNModule(_make_problem(), hp)
        result = module.configure_optimizers()
        assert isinstance(result, torch.optim.Adam)

    def test_configure_optimizers_with_lbfgs_config(self):
        hp = _make_hp(optimizer=LBFGSConfig(lr=0.5, max_iter=10))
        module = PINNModule(_make_problem(), hp)
        result = module.configure_optimizers()
        assert isinstance(result, torch.optim.LBFGS)

    def test_configure_optimizers_with_cosine_annealing(self):
        sch = CosineAnnealingConfig(T_max=100)
        hp = _make_hp(scheduler=sch)
        module = PINNModule(_make_problem(), hp)
        result = module.configure_optimizers()
        assert isinstance(result, dict)
        assert "optimizer" in result
        assert "lr_scheduler" in result

    def test_training_step_returns_tensor(self):
        hp = _make_hp()
        problem = _make_problem()
        x = torch.linspace(0, 10, 50).unsqueeze(-1)
        y = torch.randn(50, 1)
        problem.inject_context(InferredContext(x, y, {}))

        PINNModule(problem, hp)  # ensure module can be instantiated

        x_data = torch.linspace(0, 5, 10).unsqueeze(-1)
        y_data = torch.randn(10, 1)
        x_coll = torch.rand(20, 1) * 5
        batch = ((x_data, y_data), x_coll)

        # Call training_loss directly to avoid Lightning self.log() requiring a Trainer
        loss = problem.training_loss(batch)
        assert isinstance(loss, Tensor)
        assert loss.shape == ()

    def test_predict_step_output(self):
        hp = _make_hp()
        problem = _make_problem()
        x = torch.linspace(0, 10, 50).unsqueeze(-1)
        y = torch.randn(50, 1)
        problem.inject_context(InferredContext(x, y, {}))

        module = PINNModule(problem, hp)

        x_data = torch.linspace(0, 5, 10).unsqueeze(-1)
        y_data = torch.randn(10, 1)

        result = module.predict_step((x_data, y_data), 0)
        _data_batch, preds, true_vals = result
        assert "u" in preds
        assert "alpha" in preds
        assert true_vals is None  # no validation configured
