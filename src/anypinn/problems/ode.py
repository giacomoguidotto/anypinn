from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Protocol, TypeAlias, cast, override

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
    build_criterion,
)
from anypinn.lib.diff import grad as diff_grad


class ODECallable(Protocol):
    """
    Protocol for ODE right-hand side callables.

    **First-order** callables (``ODEProperties.order == 1``) receive three
    positional arguments and must match this Protocol exactly::

        def my_ode(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor: ...

    **Higher-order** callables (``order >= 2``) receive a fourth positional
    argument ``derivs: list[Tensor]``, where ``derivs[k]`` is the
    ``(k+1)``-th derivative of all fields stacked as ``(n_fields, m, 1)``::

        def my_ode(x: Tensor, y: Tensor, args: ArgsRegistry,
                   derivs: list[Tensor] = []) -> Tensor: ...

    The Protocol is intentionally kept to three arguments so that existing
    first-order callables remain valid ``ODECallable`` implementations.
    ``ResidualsConstraint`` uses ``_ODECallableN`` internally to call
    higher-order functions with the correct signature.
    """

    def __call__(self, x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor: ...


# Internal type alias for higher-order ODE callables (order >= 2).
# Not part of the public API; used only for the typed cast inside
# ResidualsConstraint.loss so that the 4-arg call is type-correct without
# a suppression comment.
_ODECallableN: TypeAlias = Callable[[Tensor, Tensor, ArgsRegistry, list[Tensor]], Tensor]


@dataclass
class ODEProperties:
    """
    Properties defining an Ordinary Differential Equation problem.

    Attributes:
        ode: The ODE function (callable).
        args: Arguments/Parameters for the ODE.
        y0: Initial conditions.
        order: Order of the ODE (default 1). For order=n, the ODE callable receives
            derivs as its last argument: derivs[k] is the (k+1)-th derivative.
        dy0: Initial conditions for lower-order derivatives, length = order-1.
            dy0[k] is the IC for the (k+1)-th derivative, shape (n_fields,).
        expected_args: Optional set of arg keys the ODE function accesses.
            When provided, validated against the merged args+params at construction time.
    """

    ode: ODECallable
    args: ArgsRegistry
    y0: Tensor
    order: int = 1
    dy0: list[Tensor] = dc_field(default_factory=list)
    expected_args: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if self.order < 1:
            raise ValueError(f"order must be >= 1, got {self.order}")
        if len(self.dy0) != self.order - 1:
            raise ValueError(f"dy0 must have length order-1={self.order - 1}, got {len(self.dy0)}")


class ResidualsConstraint(Constraint):
    """
    Constraint enforcing the ODE residuals.
    Minimizes $\\lVert \\partial y / \\partial t - f(t, y) \\rVert^2$.

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
        self.order = props.order

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

        n_fields = len(self.fields)
        x_copies = [x_coll.detach().clone().requires_grad_(True) for _ in range(n_fields)]
        preds = [f(x_copies[i]) for i, f in enumerate(self.fields.values())]
        y = torch.stack(preds)

        # Build all derivative levels by chaining (each level differentiates the previous)
        # deriv_levels[k][i] = (k+1)-th derivative of field i
        deriv_levels: list[list[Tensor]] = []
        currents = list(preds)
        for _ in range(self.order):
            next_level = [diff_grad(currents[i], x_copies[i]) for i in range(n_fields)]
            deriv_levels.append(next_level)
            currents = next_level

        # derivs[k] = (k+1)-th derivative stacked across fields, passed to the ODE callable
        derivs = [torch.stack(deriv_levels[k]) for k in range(self.order - 1)]
        # The order-th derivative is the LHS to compare against f_out
        high_deriv = torch.stack(deriv_levels[self.order - 1])

        if self.order == 1:
            f_out = self.ode(x_coll, y, self.args)
        else:
            f_out = cast(_ODECallableN, self.ode)(x_coll, y, self.args, derivs)

        loss: Tensor = self.weight * criterion(high_deriv, f_out)

        if log is not None:
            log("loss/res", loss)

        return loss


class ICConstraint(Constraint):
    """
    Constraint enforcing Initial Conditions (IC).
    Minimizes $\\lVert y(t_0) - Y_0 \\rVert^2$.

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
        self.dY0 = [dy.clone().reshape(-1, 1, 1) for dy in props.dy0]
        self.order = props.order
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
            self.dY0 = [d.to(device) for d in self.dY0]

        n_fields = len(self.fields)

        if self.order == 1:
            # Fast path: no requires_grad needed, identical to original behaviour
            Y0_preds = torch.stack([f(self.t0) for f in self.fields.values()])
            loss: Tensor = self.weight * criterion(Y0_preds, self.Y0)
        else:
            x0 = self.t0.detach().requires_grad_(True)
            preds = [f(x0) for f in self.fields.values()]
            Y0_preds = torch.stack(preds)
            total = criterion(Y0_preds, self.Y0)
            # Enforce derivative ICs by chaining from previous level
            currents = list(preds)
            for k in range(self.order - 1):
                next_level = [diff_grad(currents[i], x0) for i in range(n_fields)]
                dY0_k_pred = torch.stack(next_level)  # (n_fields, 1, 1)
                total = total + criterion(dY0_k_pred, self.dY0[k])
                currents = next_level
            loss = self.weight * total

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
    Minimizes $\\lVert \\hat{{y}} - y \\rVert^2$.

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

        criterion = build_criterion(hp.criterion)

        super().__init__(
            constraints=constraints,
            criterion=criterion,
            fields=fields,
            params=params,
        )
