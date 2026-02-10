"""Shared training script boilerplate for generated projects."""


def train_py_lightning(experiment_name: str) -> str:
    return f'''\
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import signal
import sys

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from anypinn.core import LOSS_KEY
from anypinn.lightning import PINNModule, SMMAStopping
from anypinn.lightning.callbacks import (
    DataScaling,
    FormattedProgressBar,
    Metric,
    PredictionsWriter,
)

from config import CONFIG, hp
from ode import create_data_module, create_problem


def create_dir(dir: Path) -> Path:
    dir.mkdir(exist_ok=True, parents=True)
    return dir


def clean_dir(dir: Path) -> None:
    if dir.exists():
        shutil.rmtree(dir)


def format_progress_bar(key: str, value: Metric) -> Metric:
    if LOSS_KEY in key:
        return f"{{value:.2e}}"

    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="{experiment_name}")
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Load saved model and run prediction only.",
    )
    args = parser.parse_args()

    # ========================================================================
    # Setup directories
    # ========================================================================

    log_dir = Path("./logs")
    tensorboard_dir = log_dir / "tensorboard"
    csv_dir = log_dir / "csv"

    models_dir = Path("./models") / CONFIG.experiment_name / CONFIG.run_name
    model_path = models_dir / "model.ckpt"
    predictions_dir = models_dir

    temp_dir = Path("./temp")

    create_dir(log_dir)
    create_dir(models_dir)
    create_dir(temp_dir)

    clean_dir(temp_dir)
    if not args.predict:
        clean_dir(csv_dir / CONFIG.experiment_name / CONFIG.run_name)
        clean_dir(tensorboard_dir / CONFIG.experiment_name / CONFIG.run_name)

    # ========================================================================
    # Build components
    # ========================================================================

    dm = create_data_module(hp)
    problem = create_problem(hp)

    if args.predict:
        module = PINNModule.load_from_checkpoint(
            model_path,
            problem=problem,
            weights_only=False,
        )
    else:
        module = PINNModule(
            problem=problem,
            hp=hp,
        )

    callbacks = [
        ModelCheckpoint(
            dirpath=temp_dir,
            filename="{{epoch:02d}}",
            monitor=LOSS_KEY,
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        LearningRateMonitor(
            logging_interval="epoch",
        ),
        FormattedProgressBar(
            refresh_rate=10,
            format=format_progress_bar,
        ),
        PredictionsWriter(
            predictions_path=predictions_dir / "predictions.pt",
        ),
    ]

    if hp.smma_stopping:
        callbacks.append(
            SMMAStopping(
                config=hp.smma_stopping,
                loss_key=LOSS_KEY,
            ),
        )

    loggers = [
        TensorBoardLogger(
            save_dir=tensorboard_dir,
            name=CONFIG.experiment_name,
            version=CONFIG.run_name,
        ),
        CSVLogger(
            save_dir=csv_dir,
            name=CONFIG.experiment_name,
            version=CONFIG.run_name,
        ),
    ]

    trainer = Trainer(
        max_epochs=CONFIG.max_epochs,
        gradient_clip_val=CONFIG.gradient_clip_val,
        logger=loggers if not args.predict else [],
        callbacks=callbacks,
        log_every_n_steps=0,
    )

    # ========================================================================
    # Execution
    # ========================================================================

    if not args.predict:

        def on_interrupt(_signum, _frame):
            print("\\nTraining interrupted. Saving checkpoint and predictions...")
            trainer.save_checkpoint(model_path, weights_only=False)
            trainer.predict(module, dm)
            clean_dir(temp_dir)
            sys.exit(0)

        signal.signal(signal.SIGINT, on_interrupt)
        trainer.fit(module, dm)
        trainer.save_checkpoint(model_path, weights_only=False)

    trainer.predict(module, dm)
    clean_dir(temp_dir)


if __name__ == "__main__":
    main()
'''


def train_py_core(experiment_name: str) -> str:
    return f'''\
from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys

import torch

from config import CONFIG, hp
from ode import create_context, create_problem


def main() -> None:
    parser = argparse.ArgumentParser(description="{experiment_name}")
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Load saved model and run prediction only.",
    )
    args = parser.parse_args()

    # ========================================================================
    # Setup
    # ========================================================================

    models_dir = Path("./models") / CONFIG.experiment_name / CONFIG.run_name
    models_dir.mkdir(exist_ok=True, parents=True)
    model_path = models_dir / "model.pt"

    problem = create_problem(hp)
    context = create_context()
    problem.inject_context(context)

    if args.predict:
        problem.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        optimizer = torch.optim.Adam(problem.parameters(), lr=hp.lr)

        def on_interrupt(_signum, _frame):
            print("\\nTraining interrupted. Saving model...")
            torch.save(problem.state_dict(), model_path)
            sys.exit(0)

        signal.signal(signal.SIGINT, on_interrupt)

        # TODO: implement your training loop here
        # for epoch in range(CONFIG.max_epochs):
        #     for batch in your_dataloader:
        #         optimizer.zero_grad()
        #         loss = problem.training_loss(batch, log=None)
        #         loss.backward()
        #         optimizer.step()

        torch.save(problem.state_dict(), model_path)

    print("Done.")


if __name__ == "__main__":
    main()
'''
