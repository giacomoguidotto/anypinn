from __future__ import annotations

from typing import cast, override

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch
from torch import Tensor

from anypinn.core import LOSS_KEY, LogFn, Predictions, Problem, TrainingBatch
from anypinn.core.config import AdamConfig, CosineAnnealingConfig, LBFGSConfig, PINNHyperparameters
from anypinn.core.types import PredictionBatch


class PINNModule(pl.LightningModule):
    """
    Generic PINN Lightning module.
    Expects external Problem + Sampler + optimizer config.

    Args:
        problem: The PINN problem definition (constraints, fields, etc.).
        hp: Hyperparameters for training.
    """

    def __init__(
        self,
        problem: Problem,
        hp: PINNHyperparameters,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["problem"])

        self.problem = problem
        self.hp = hp

        def _log(key: str, value: Tensor, progress_bar: bool = False) -> None:
            self.log(
                key,
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=progress_bar,
                batch_size=hp.training_data.batch_size,
            )

        self._log = cast(LogFn, _log)

    @override
    def on_fit_start(self) -> None:
        """
        Called when fit begins. Resolves validation sources using loaded data.
        """
        self.problem.inject_context(self.trainer.datamodule.context)  # type: ignore

    @override
    def on_predict_start(self) -> None:
        """
        Called when predict begins. Resolves validation sources using loaded data.
        """
        self.problem.inject_context(self.trainer.datamodule.context)  # type: ignore

    @override
    def training_step(self, batch: TrainingBatch, batch_idx: int) -> Tensor:
        """
        Performs a single training step.
        Calculates total loss from the problem.
        """
        return self.problem.training_loss(batch, self._log)

    @override
    def predict_step(self, batch: PredictionBatch, batch_idx: int) -> Predictions:
        """
        Performs a prediction step.
        """
        x_data, y_data = batch

        (data_batch, predictions) = self.problem.predict((x_data, y_data))
        true_values = self.problem.true_values(x_data)

        return (data_batch, predictions, true_values)

    @override
    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Configures the optimizer and learning rate scheduler.
        """
        opt_cfg = self.hp.optimizer
        if isinstance(opt_cfg, LBFGSConfig):
            opt = torch.optim.LBFGS(
                self.parameters(),
                lr=opt_cfg.lr,
                max_iter=opt_cfg.max_iter,
                max_eval=opt_cfg.max_eval,
                history_size=opt_cfg.history_size,
                line_search_fn=opt_cfg.line_search_fn,
            )
        elif isinstance(opt_cfg, AdamConfig):
            opt = torch.optim.Adam(
                self.parameters(),
                lr=opt_cfg.lr,
                betas=opt_cfg.betas,
                weight_decay=opt_cfg.weight_decay,
            )
        else:
            opt = torch.optim.Adam(self.parameters(), lr=self.hp.lr)

        sch_cfg = self.hp.scheduler
        if not sch_cfg:
            return opt

        if isinstance(sch_cfg, CosineAnnealingConfig):
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=sch_cfg.T_max,
                eta_min=sch_cfg.eta_min,
            )
            return {
                "optimizer": opt,
                "lr_scheduler": {
                    "name": "lr",
                    "scheduler": sch,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=sch_cfg.mode,
            factor=sch_cfg.factor,
            patience=sch_cfg.patience,
            threshold=sch_cfg.threshold,
            min_lr=sch_cfg.min_lr,
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "name": "lr",
                "scheduler": sch,
                "monitor": LOSS_KEY,
                "interval": "epoch",
                "frequency": 1,
            },
        }
