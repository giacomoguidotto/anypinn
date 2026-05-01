"""Core problem abstractions for PINN."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor
import torch.nn as nn

from anypinn.core.context import InferredContext
from anypinn.core.nn import FieldsRegistry, Parameter, ParamsRegistry
from anypinn.core.types import LOSS_KEY, DataBatch, LogFn, TrainingBatch
from anypinn.core.validation import _ColumnLookup


class Constraint(ABC):
    """
    Abstract base class for a constraint (loss term) in the PINN.

    Subclass this and implement ``loss()`` to define custom physics or
    data-fitting terms. The ``Problem`` sums all constraint losses during
    training.

    Example:
        ```python
        class EnergyConstraint(Constraint):
            def loss(self, batch, criterion, log=None):
                (x_data, y_data), x_coll = batch
                energy = compute_energy(x_coll)
                target = torch.zeros_like(energy)
                loss = criterion(energy, target)
                if log is not None:
                    log("loss/energy", loss)
                return loss
        ```
    """

    def inject_context(self, context: InferredContext) -> None:
        """
        Inject the context into the constraint. This can be used by the constraint to access the
        data used to compute the loss.

        Args:
            context: The context to inject.
        """
        return None

    @abstractmethod
    def loss(
        self,
        batch: TrainingBatch,
        criterion: nn.Module,
        log: LogFn | None = None,
    ) -> Tensor:
        """
        Calculate the loss for this constraint.

        Args:
            batch: The current batch of data/collocation points.
            criterion: The loss function (e.g. MSE).
            log: Optional logging function.

        Returns:
            The calculated loss tensor.
        """


class Problem(nn.Module):
    """
    Aggregates constraints into a total training loss.

    Manages fields (neural networks), learnable parameters, and the loss
    criterion. Call ``training_loss()`` during each training step and
    ``predict()`` for inference.

    Args:
        constraints: List of constraints to enforce.
        criterion: Loss function module.
        fields: Registry of named neural fields.
        params: Registry of named learnable parameters.

    Example:
        ```python
        problem = Problem(
            constraints=[residual_constraint, ic_constraint],
            criterion=nn.MSELoss(),
            fields={"u": field},
            params={"alpha": Parameter(ScalarConfig(init_value=0.01))},
        )
        ```
    """

    def __init__(
        self,
        constraints: list[Constraint],
        criterion: nn.Module,
        fields: FieldsRegistry,
        params: ParamsRegistry,
    ):
        super().__init__()
        self.constraints = constraints
        self.criterion = criterion
        self.fields = fields
        self.params = params

        self._fields = nn.ModuleList(fields.values())
        self._params = nn.ModuleList(params.values())

    def inject_context(self, context: InferredContext) -> None:
        """
        Inject the context into the problem.

        This should be called after data is loaded but before training starts.
        Pure function entries are passed through unchanged.

        Args:
            context: The context to inject.
        """
        self.context = context
        for c in self.constraints:
            c.inject_context(context)

    def training_loss(self, batch: TrainingBatch, log: LogFn | None = None) -> Tensor:
        """
        Calculate the total loss from all constraints.

        Args:
            batch: Current batch.
            log: Optional logging function.

        Returns:
            Sum of losses from all constraints.
        """
        _, x_coll = batch

        if not self.constraints:
            total = torch.tensor(0.0, device=x_coll.device)
        else:
            losses = iter(self.constraints)
            total = next(losses).loss(batch, self.criterion, log)
            for c in losses:
                total = total + c.loss(batch, self.criterion, log)

        if log is not None:
            for name, param in self.params.items():
                param_loss = self._param_validation_loss(name, param, x_coll)
                if param_loss is not None:
                    log(f"loss/{name}", param_loss, progress_bar=True)

            log(LOSS_KEY, total, progress_bar=True)

        return total

    def predict(self, batch: DataBatch) -> tuple[DataBatch, dict[str, Tensor]]:
        """
        Generate predictions for a given batch of data.
        Returns unscaled predictions in original domain.

        Args:
            batch: Batch of input coordinates.

        Returns:
            Tuple of (original_batch, predictions_dict).
        """

        x, y = batch

        n = x.shape[0]
        preds = {name: f(x).reshape(n, -1).squeeze(-1) for name, f in self.fields.items()}
        preds |= {name: p(x).reshape(n, -1).squeeze(-1) for name, p in self.params.items()}

        return (x.squeeze(-1), y.squeeze(-1)), preds

    def true_values(self, x: Tensor) -> dict[str, Tensor] | None:
        """
        Get the true values for a given x coordinates.
        Returns None if no validation source is configured.
        """

        return {
            name: p_true.reshape(x.shape[0], -1).squeeze(-1)
            for name, p in self.params.items()
            if (p_true := self._get_true_param(name, x)) is not None
        } or None

    def _get_true_param(self, param_name: str, x: Tensor) -> Tensor | None:
        """
        Get the ground truth values for a parameter at given coordinates.

        Args:
            param_name: Name of the parameter.
            x: Input coordinates.

        Returns:
            Ground truth values, or None if no validation source is configured.
        """
        if param_name not in self.context.validation:
            return None

        fn = self.context.validation[param_name]

        if isinstance(fn, _ColumnLookup):
            domain = self.context.domain
            if domain.dx is None:
                raise ValueError(
                    f"Cannot perform ColumnRef lookup for '{param_name}': "
                    "domain step size (dx) is unknown. Ensure the domain was inferred from "
                    "a uniformly-spaced coordinate tensor, or use a callable validation source."
                )
            x_idx = ((x.squeeze(-1) - domain.x0) / domain.dx[0]).round().unsqueeze(-1)
            return fn(x_idx)

        return fn(x)

    @torch.no_grad()
    def _param_validation_loss(
        self, param_name: str, param: Parameter, x_coll: Tensor
    ) -> Tensor | None:
        """
        Compute validation loss for a parameter against ground truth.

        Args:
            param: The parameter to compute validation loss for.
            x_coll: The input coordinates.

        Returns:
            Loss value, or None if no validation source is configured.
        """
        true = self._get_true_param(param_name, x_coll)
        if true is None:
            return None

        pred = param(x_coll)

        return torch.mean((true - pred) ** 2)
