from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeAlias, override

import torch
from torch import Tensor
import torch.nn as nn

from anypinn.core import (
    ArgsRegistry,
    Constraint,
    FieldsRegistry,
    InferredContext,
    LogFn,
    ParamsRegistry,
    PINNHyperparameters,
    Problem,
    TrainingBatch,
)


class ODECallable(Protocol):
    def __call__(
        self,
        x: Tensor,
        y: Tensor,
        args: ArgsRegistry,
    ) -> Tensor: ...


@dataclass
class ODEProperties:
    """
    Properties defining an Ordinary Differential Equation problem.

    Attributes:
        ode: The ODE function (callable).
        args: Arguments/Parameters for the ODE.
        y0: Initial conditions.
        expected_args: Optional set of arg keys the ODE function accesses.
            When provided, validated against the merged args+params at construction time.
    """

    ode: ODECallable
    args: ArgsRegistry
    y0: Tensor
    expected_args: frozenset[str] | None = None


class ResidualsConstraint(Constraint):
    """
    Constraint enforcing the ODE residuals.
    Minimizes ||dy/dt - f(t, y)||^2.

    Args:
        props: ODE properties.
        fields: List of fields.
        params: List of parameters.
        weight: Weight for this loss term.
    """

    def __init__(
        self,
        props: ODEProperties,
        fields: FieldsRegistry,
        params: ParamsRegistry,
        weight: float = 1.0,
    ):
        if len(fields) != len(props.y0):
            raise ValueError(
                f"Number of fields ({len(fields)}) must match number of initial conditions "
                f"in y0 ({len(props.y0)}). Field keys: {list(fields)}."
            )

        merged_args: dict[str, object] = {**props.args, **params}
        if props.expected_args is not None:
            missing = props.expected_args - merged_args.keys()
            if missing:
                raise ValueError(
                    f"ODE function expects args {sorted(missing)!r} but they are not in "
                    f"props.args or params. Available keys: {sorted(merged_args.keys())!r}."
                )

        self.fields = fields
        self.weight = weight

        self.ode = props.ode

        # add the trainable params as args
        self.args = props.args.copy()
        self.args.update(params)

    @override
    def loss(
        self,
        batch: TrainingBatch,
        criterion: nn.Module,
        log: LogFn | None = None,
    ) -> Tensor:
        _, x_coll = batch

        n = len(self.fields)
        x_copies = [x_coll.detach().clone().requires_grad_(True) for _ in range(n)]
        preds = [f(x_copies[i]) for i, f in enumerate(self.fields.values())]
        y = torch.stack(preds)

        dy_dt_pred = self.ode(x_coll, y, self.args)

        ones = torch.ones_like(preds[0])
        dy_dt = torch.stack(torch.autograd.grad(preds, x_copies, [ones] * n, create_graph=True))

        loss: Tensor = self.weight * criterion(dy_dt, dy_dt_pred)

        if log is not None:
            log("loss/res", loss)

        return loss


class ICConstraint(Constraint):
    """
    Constraint enforcing Initial Conditions (IC).
    Minimizes ||y(t0) - Y0||^2.

    Args:
        fields: Fields registry.
        weight: Weight for this loss term.
    """

    def __init__(
        self,
        props: ODEProperties,
        fields: FieldsRegistry,
        weight: float = 1.0,
    ):
        if len(fields) != len(props.y0):
            raise ValueError(
                f"Number of fields ({len(fields)}) must match number of initial conditions "
                f"in y0 ({len(props.y0)}). Field keys: {list(fields)}."
            )

        self.Y0 = props.y0.clone().reshape(-1, 1, 1)
        self.fields = fields
        self.weight = weight

    @override
    def inject_context(self, context: InferredContext) -> None:
        """
        Inject the context into the constraint.
        """
        self.t0 = torch.tensor(context.domain.x0, dtype=torch.float32).reshape(1, 1)

    @override
    def loss(
        self,
        batch: TrainingBatch,
        criterion: nn.Module,
        log: LogFn | None = None,
    ) -> Tensor:
        device = batch[1].device

        if self.t0.device != device:
            self.t0 = self.t0.to(device)
            self.Y0 = self.Y0.to(device)

        Y0_preds = torch.stack([f(self.t0) for f in self.fields.values()])

        loss: Tensor = criterion(Y0_preds, self.Y0)
        loss = self.weight * loss

        if log is not None:
            log("loss/ic", loss)

        return loss


PredictDataFn: TypeAlias = Callable[[Tensor, FieldsRegistry, ParamsRegistry], Tensor]


@dataclass(kw_only=True)
class ODEHyperparameters(PINNHyperparameters):
    """
    Hyperparameters for ODE inverse problems.
    """

    pde_weight: float = 1.0
    ic_weight: float = 1.0
    data_weight: float = 1.0


class DataConstraint(Constraint):
    """
    Constraint enforcing fit to observed data.
    Minimizes ||Predictions - Data||^2.

    Args:
        fields: Fields registry.
        params: Parameters registry.
        predict_data: Function to predict data values from fields.
        weight: Weight for this loss term.
    """

    def __init__(
        self,
        fields: FieldsRegistry,
        params: ParamsRegistry,
        predict_data: PredictDataFn,
        weight: float = 1.0,
    ):
        self.fields = fields
        self.params = params
        self.predict_data = predict_data
        self.weight = weight

    @override
    def loss(
        self,
        batch: TrainingBatch,
        criterion: nn.Module,
        log: LogFn | None = None,
    ) -> Tensor:
        (x_data, y_data), _ = batch

        y_data_pred = self.predict_data(x_data, self.fields, self.params)

        loss: Tensor = criterion(y_data_pred, y_data)
        loss = self.weight * loss

        if log is not None:
            log("loss/data", loss)

        return loss


class ODEInverseProblem(Problem):
    """
    Generic ODE Inverse Problem.
    Composes Residuals + IC + Data constraints with MSELoss.
    """

    def __init__(
        self,
        props: ODEProperties,
        hp: ODEHyperparameters,
        fields: FieldsRegistry,
        params: ParamsRegistry,
        predict_data: PredictDataFn,
    ) -> None:
        constraints: list[Constraint] = [
            ResidualsConstraint(
                props=props,
                fields=fields,
                params=params,
                weight=hp.pde_weight,
            ),
            ICConstraint(
                props=props,
                fields=fields,
                weight=hp.ic_weight,
            ),
            DataConstraint(
                fields=fields,
                params=params,
                predict_data=predict_data,
                weight=hp.data_weight,
            ),
        ]

        criterion = nn.MSELoss()

        super().__init__(
            constraints=constraints,
            criterion=criterion,
            fields=fields,
            params=params,
        )
