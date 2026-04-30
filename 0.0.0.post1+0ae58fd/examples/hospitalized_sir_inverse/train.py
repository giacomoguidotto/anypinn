from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import signal
import sys

from config import EXPERIMENT_NAME, hp
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from ode import C_H, C_I, create_data_module, create_problem, plot_and_save, props

from anypinn.core import LOSS_KEY
from anypinn.lightning import PINNModule, SMMAStopping
from anypinn.lightning.callbacks import FormattedProgressBar, Metric, PredictionsWriter

# ============================================================================
# Helpers
# ============================================================================


def create_dir(dir: Path) -> Path:
    dir.mkdir(exist_ok=True, parents=True)
    return dir


def clean_dir(dir: Path) -> None:
    if dir.exists():
        shutil.rmtree(dir)


def format_progress_bar(key: str, value: Metric) -> Metric:
    if LOSS_KEY in key:
        return f"{value:.2e}"

    return value


def main(experiment_name: str, predict: bool = False) -> None:
    log_dir = Path("../_logs")
    tensorboard_dir = log_dir / "tensorboard"
    csv_dir = log_dir / "csv"

    results_dir = Path("./results")
    temp_dir = Path("./temp")

    results_dir.mkdir(exist_ok=True, parents=True)
    create_dir(log_dir)
    create_dir(temp_dir)

    clean_dir(temp_dir)
    if not predict:
        clean_dir(csv_dir / experiment_name)
        clean_dir(tensorboard_dir / experiment_name)

    model_path = results_dir / "model.ckpt"

    dm = create_data_module(hp)
    problem = create_problem(hp)

    if predict:
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
            filename="{epoch:02d}",
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
            predictions_path=results_dir / f"{experiment_name}.pt",
            on_prediction=lambda _, __, predictions_list, ___: plot_and_save(
                predictions_list[0], results_dir, experiment_name, props, C_I, C_H
            ),
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
            name=experiment_name,
            version="",
        ),
        CSVLogger(
            save_dir=csv_dir,
            name=experiment_name,
            version="",
        ),
    ]

    trainer = Trainer(
        max_epochs=hp.max_epochs,
        gradient_clip_val=hp.gradient_clip_val,
        logger=loggers if not predict else [],
        callbacks=callbacks,
        log_every_n_steps=0,
    )

    # ============================================================================
    # Execution
    # ============================================================================

    if not predict:

        def on_interrupt(_signum, _frame):
            print("\nTraining interrupted. Saving checkpoint and predictions...")
            trainer.save_checkpoint(model_path, weights_only=False)
            trainer.predict(module, dm)
            clean_dir(temp_dir)
            sys.exit(0)

        signal.signal(signal.SIGINT, on_interrupt)
        trainer.fit(module, dm)
        trainer.save_checkpoint(model_path, weights_only=False)

    trainer.predict(module, dm)

    clean_dir(temp_dir)


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIR Inverse Problem using Hospitalization Data")
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Load saved model and run prediction. Does not train the model.",
    )
    args = parser.parse_args()

    main(EXPERIMENT_NAME, args.predict)
