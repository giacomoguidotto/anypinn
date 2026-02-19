"""Tests for anypinn.problems.ode — ODE constraints and ODEInverseProblem."""

import pytest
import torch
from torch import Tensor
import torch.nn as nn

from anypinn.core.config import MLPConfig, ScalarConfig
from anypinn.core.context import InferredContext
from anypinn.core.nn import Argument, Field, Parameter
from anypinn.problems.ode import (
    DataConstraint,
    ICConstraint,
    ODEHyperparameters,
    ODEInverseProblem,
    ODEProperties,
    ResidualsConstraint,
)


def _decay_ode(x: Tensor, y: Tensor, args: dict[str, Argument]) -> Tensor:
    """Simple decay ODE: dy/dt = -y."""
    return -y


def _make_fields(n_fields: int = 1) -> dict[str, Field]:
    cfg = MLPConfig(in_dim=1, out_dim=1, hidden_layers=[8], activation="tanh")
    return {f"u{i}": Field(cfg) for i in range(n_fields)}


def _make_params() -> dict[str, Parameter]:
    return {"alpha": Parameter(ScalarConfig(init_value=0.5))}


def _make_props(n_fields: int = 1) -> ODEProperties:
    return ODEProperties(
        ode=_decay_ode,
        args={"k": Argument(1.0)},
        y0=torch.zeros(n_fields),
    )


class TestResidualsConstraint:
    def test_loss_computes(self):
        fields = _make_fields(1)
        params = _make_params()
        props = _make_props(1)
        constraint = ResidualsConstraint(props, fields, params, weight=1.0)

        x_data = torch.linspace(0, 5, 10).unsqueeze(-1)
        y_data = torch.randn(10, 1)
        x_coll = torch.linspace(0, 5, 20).unsqueeze(-1)
        batch = ((x_data, y_data), x_coll)

        loss = constraint.loss(batch, nn.MSELoss())
        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_weight_scales_loss(self):
        fields = _make_fields(1)
        params = _make_params()
        props = _make_props(1)

        c1 = ResidualsConstraint(props, fields, params, weight=1.0)
        c2 = ResidualsConstraint(props, fields, params, weight=2.0)

        x_data = torch.linspace(0, 5, 10).unsqueeze(-1)
        y_data = torch.randn(10, 1)
        x_coll = torch.linspace(0, 5, 20).unsqueeze(-1)
        batch = ((x_data, y_data), x_coll)

        loss1 = c1.loss(batch, nn.MSELoss())
        loss2 = c2.loss(batch, nn.MSELoss())
        assert loss2.item() == pytest.approx(loss1.item() * 2.0, rel=1e-5)

    def test_loss_multi_field(self):
        fields = _make_fields(2)
        params = _make_params()
        props = _make_props(2)
        constraint = ResidualsConstraint(props, fields, params, weight=1.0)

        x_data = torch.linspace(0, 5, 10).unsqueeze(-1)
        y_data = torch.randn(10, 1)
        x_coll = torch.linspace(0, 5, 20).unsqueeze(-1)
        batch = ((x_data, y_data), x_coll)

        loss = constraint.loss(batch, nn.MSELoss())
        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_loss_with_logging(self):
        fields = _make_fields(1)
        params = _make_params()
        props = _make_props(1)
        constraint = ResidualsConstraint(props, fields, params)

        x_data = torch.linspace(0, 5, 10).unsqueeze(-1)
        y_data = torch.randn(10, 1)
        x_coll = torch.linspace(0, 5, 20).unsqueeze(-1)
        batch = ((x_data, y_data), x_coll)

        logged = {}

        def log_fn(name: str, value: Tensor, progress_bar: bool = False) -> None:
            logged[name] = value

        constraint.loss(batch, nn.MSELoss(), log=log_fn)
        assert "loss/res" in logged


class TestICConstraint:
    def test_loss_computes(self):
        fields = _make_fields(1)
        props = _make_props(1)
        constraint = ICConstraint(props, fields, weight=1.0)

        # Must inject context to get t0
        x = torch.linspace(0, 10, 50).unsqueeze(-1)
        y = torch.randn(50, 1)
        ctx = InferredContext(x, y, {})
        constraint.inject_context(ctx)

        x_data = torch.linspace(0, 5, 10).unsqueeze(-1)
        y_data = torch.randn(10, 1)
        x_coll = torch.linspace(0, 5, 20).unsqueeze(-1)
        batch = ((x_data, y_data), x_coll)

        loss = constraint.loss(batch, nn.MSELoss())
        assert loss.shape == ()
        assert loss.item() >= 0.0

    def test_weight_scales_loss(self):
        fields = _make_fields(1)
        props = _make_props(1)

        c1 = ICConstraint(props, fields, weight=1.0)
        c2 = ICConstraint(props, fields, weight=3.0)

        x = torch.linspace(0, 10, 50).unsqueeze(-1)
        y = torch.randn(50, 1)
        ctx = InferredContext(x, y, {})
        c1.inject_context(ctx)
        c2.inject_context(ctx)

        x_data = torch.linspace(0, 5, 10).unsqueeze(-1)
        y_data = torch.randn(10, 1)
        x_coll = torch.linspace(0, 5, 20).unsqueeze(-1)
        batch = ((x_data, y_data), x_coll)

        loss1 = c1.loss(batch, nn.MSELoss())
        loss3 = c2.loss(batch, nn.MSELoss())
        assert loss3.item() == pytest.approx(loss1.item() * 3.0, rel=1e-5)


class TestDataConstraint:
    def test_loss_computes(self):
        fields = _make_fields(1)
        params = _make_params()

        def predict_data(x: Tensor, fields: dict, params: dict) -> Tensor:
            return fields["u0"](x)

        constraint = DataConstraint(fields, params, predict_data, weight=1.0)

        x_data = torch.linspace(0, 5, 10).unsqueeze(-1)
        y_data = torch.randn(10, 1)
        x_coll = torch.linspace(0, 5, 20).unsqueeze(-1)
        batch = ((x_data, y_data), x_coll)

        loss = constraint.loss(batch, nn.MSELoss())
        assert loss.shape == ()
        assert loss.item() >= 0.0


class TestODEInverseProblem:
    def test_composition(self):
        fields = _make_fields(1)
        params = _make_params()
        props = _make_props(1)

        hp = ODEHyperparameters(
            lr=1e-3,
            training_data=pytest.importorskip("anypinn.core.config").GenerationConfig(
                batch_size=16,
                data_ratio=0.5,
                collocations=50,
                x=torch.linspace(0, 10, 50),
                noise_level=0.0,
                args_to_train={},
            ),
            fields_config=MLPConfig(in_dim=1, out_dim=1, hidden_layers=[8], activation="tanh"),
            params_config=ScalarConfig(init_value=0.5),
            pde_weight=1.0,
            ic_weight=2.0,
            data_weight=3.0,
        )

        def predict_data(x: Tensor, f: dict, p: dict) -> Tensor:
            return f["u0"](x)

        problem = ODEInverseProblem(props, hp, fields, params, predict_data)

        assert len(problem.constraints) == 3
        assert isinstance(problem.constraints[0], ResidualsConstraint)
        assert isinstance(problem.constraints[1], ICConstraint)
        assert isinstance(problem.constraints[2], DataConstraint)

    def test_training_loss_runs(self):
        fields = _make_fields(1)
        params = _make_params()
        props = _make_props(1)

        from anypinn.core.config import GenerationConfig

        hp = ODEHyperparameters(
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
        )

        def predict_data(x: Tensor, f: dict, p: dict) -> Tensor:
            return f["u0"](x)

        problem = ODEInverseProblem(props, hp, fields, params, predict_data)

        x = torch.linspace(0, 10, 50).unsqueeze(-1)
        y = torch.randn(50, 1)
        ctx = InferredContext(x, y, {})
        problem.inject_context(ctx)

        x_data = torch.linspace(0, 5, 10).unsqueeze(-1)
        y_data = torch.randn(10, 1)
        x_coll = torch.linspace(0, 5, 20).unsqueeze(-1)
        batch = ((x_data, y_data), x_coll)

        loss = problem.training_loss(batch)
        assert loss.shape == ()
        assert loss.item() >= 0.0
