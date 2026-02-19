"""Tests for anypinn.core.problem — Problem and Constraint."""

import torch
from torch import Tensor
import torch.nn as nn

from anypinn.core.config import MLPConfig, ScalarConfig
from anypinn.core.context import InferredContext
from anypinn.core.nn import Field, Parameter
from anypinn.core.problem import Constraint, Problem
from anypinn.core.types import LogFn, TrainingBatch
from anypinn.core.validation import ResolvedValidation


class DummyConstraint(Constraint):
    """Returns a fixed loss value for testing."""

    def __init__(self, value: float = 1.0):
        self._value = value
        self.context_injected = False

    def inject_context(self, context: InferredContext) -> None:
        self.context_injected = True

    def loss(
        self,
        batch: TrainingBatch,
        criterion: nn.Module,
        log: LogFn | None = None,
    ) -> Tensor:
        return torch.tensor(self._value)


def _make_problem(
    constraints: list[Constraint] | None = None,
    fields: dict[str, Field] | None = None,
    params: dict[str, Parameter] | None = None,
) -> Problem:
    if fields is None:
        cfg = MLPConfig(in_dim=1, out_dim=1, hidden_layers=[8], activation="tanh")
        fields = {"u": Field(cfg)}
    if params is None:
        params = {"alpha": Parameter(ScalarConfig(init_value=0.5))}
    if constraints is None:
        constraints = [DummyConstraint(1.0)]
    return Problem(constraints=constraints, criterion=nn.MSELoss(), fields=fields, params=params)


class TestProblem:
    def test_training_loss_single_constraint(self, training_batch: TrainingBatch):
        problem = _make_problem(constraints=[DummyConstraint(2.5)])
        x_tensor = torch.linspace(0, 10, 50).unsqueeze(-1)
        y_tensor = torch.randn(50, 1)
        ctx = InferredContext(x_tensor, y_tensor, {})
        problem.inject_context(ctx)

        loss = problem.training_loss(training_batch)
        assert loss.item() == 2.5

    def test_training_loss_sums_constraints(self, training_batch: TrainingBatch):
        c1, c2, c3 = DummyConstraint(1.0), DummyConstraint(2.0), DummyConstraint(3.0)
        problem = _make_problem(constraints=[c1, c2, c3])
        x_tensor = torch.linspace(0, 10, 50).unsqueeze(-1)
        y_tensor = torch.randn(50, 1)
        problem.inject_context(InferredContext(x_tensor, y_tensor, {}))

        loss = problem.training_loss(training_batch)
        assert loss.item() == 6.0

    def test_training_loss_empty_constraints(self, training_batch: TrainingBatch):
        problem = _make_problem(constraints=[])
        x_tensor = torch.linspace(0, 10, 50).unsqueeze(-1)
        y_tensor = torch.randn(50, 1)
        problem.inject_context(InferredContext(x_tensor, y_tensor, {}))

        loss = problem.training_loss(training_batch)
        assert loss.item() == 0.0

    def test_training_loss_with_log(self, training_batch: TrainingBatch):
        problem = _make_problem()
        x_tensor = torch.linspace(0, 10, 50).unsqueeze(-1)
        y_tensor = torch.randn(50, 1)
        problem.inject_context(InferredContext(x_tensor, y_tensor, {}))

        logged: dict[str, object] = {}

        def log_fn(name: str, value: Tensor, progress_bar: bool = False) -> None:
            logged[name] = value

        problem.training_loss(training_batch, log=log_fn)
        assert "loss" in logged

    def test_context_injection_propagates(self):
        c1, c2 = DummyConstraint(), DummyConstraint()
        problem = _make_problem(constraints=[c1, c2])
        x_tensor = torch.linspace(0, 10, 50).unsqueeze(-1)
        y_tensor = torch.randn(50, 1)
        ctx = InferredContext(x_tensor, y_tensor, {})
        problem.inject_context(ctx)

        assert c1.context_injected
        assert c2.context_injected
        assert problem.context is ctx

    def test_predict_output_structure(self):
        cfg = MLPConfig(in_dim=1, out_dim=1, hidden_layers=[8], activation="tanh")
        fields = {"u": Field(cfg)}
        params = {"alpha": Parameter(ScalarConfig(init_value=0.5))}
        problem = _make_problem(fields=fields, params=params)

        x = torch.linspace(0, 5, 20).unsqueeze(-1)
        y = torch.randn(20, 1)

        (x_out, y_out), preds = problem.predict((x, y))

        assert x_out.shape == (20,)
        assert y_out.shape == (20,)
        assert "u" in preds
        assert "alpha" in preds
        assert preds["u"].shape == (20,)

    def test_true_values_with_validation(self):
        validation: ResolvedValidation = {
            "alpha": lambda x: torch.ones_like(x.squeeze(-1)) * 0.3,
        }
        problem = _make_problem()
        x_tensor = torch.linspace(0, 10, 50).unsqueeze(-1)
        y_tensor = torch.randn(50, 1)
        problem.inject_context(InferredContext(x_tensor, y_tensor, validation))

        x = torch.linspace(0, 5, 10).unsqueeze(-1)
        tv = problem.true_values(x)
        assert tv is not None
        assert "alpha" in tv

    def test_true_values_without_validation(self):
        problem = _make_problem()
        x_tensor = torch.linspace(0, 10, 50).unsqueeze(-1)
        y_tensor = torch.randn(50, 1)
        problem.inject_context(InferredContext(x_tensor, y_tensor, {}))

        x = torch.linspace(0, 5, 10).unsqueeze(-1)
        tv = problem.true_values(x)
        assert tv is None
