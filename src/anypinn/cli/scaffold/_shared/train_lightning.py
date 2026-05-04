from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import shutil
import signal
import sys
import webbrowser

from config import EXPERIMENT_NAME, hp
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from ode import create_data_module, create_problem, plot_and_save

from anypinn.core import LOSS_KEY
from anypinn.lightning import PINNModule, SMMAStopping
from anypinn.lightning.callbacks import FormattedProgressBar, Metric, PredictionsWriter


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


def _next_version(base: Path) -> int:
    if not base.exists():
        return 0
    versions = [
        int(d.name.split("_")[1])
        for d in base.iterdir()
        if d.is_dir() and d.name.startswith("version_")
    ]
    return max(versions) + 1 if versions else 0


def _latest_version(base: Path) -> int:
    if not base.exists():
        raise FileNotFoundError("No results found. Train a model first.")
    versions = [
        int(d.name.split("_")[1])
        for d in base.iterdir()
        if d.is_dir() and d.name.startswith("version_")
    ]
    if not versions:
        raise FileNotFoundError("No results found. Train a model first.")
    return max(versions)


def main() -> None:
    parser = argparse.ArgumentParser(description="__EXPERIMENT_NAME__")
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Load saved model and run prediction only.",
    )
    args = parser.parse_args()

    # ========================================================================
    # Setup directories
    # ========================================================================

    results_dir = Path("./results")
    log_dir = Path("./logs")
    temp_dir = Path("./temp")

    version = _latest_version(results_dir) if args.predict else _next_version(results_dir)
    version_name = f"version_{version}"
    run_dir = create_dir(results_dir / version_name)
    model_path = run_dir / "model.ckpt"
    plot_path = run_dir / "plot.png"

    create_dir(log_dir)
    create_dir(temp_dir)

    clean_dir(temp_dir)
    if not args.predict:
        clean_dir(log_dir / "csv" / version_name)
        clean_dir(log_dir / "tensorboard" / version_name)

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
            predictions_path=run_dir / "predictions.pt",
            on_prediction=lambda _, __, predictions_list, ___: plot_and_save(
                predictions_list[0], run_dir, EXPERIMENT_NAME
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
            save_dir=log_dir / "tensorboard",
            name="",
            version=version_name,
        ),
        CSVLogger(
            save_dir=log_dir / "csv",
            name="",
            version=version_name,
        ),
    ]

    trainer = Trainer(
        max_epochs=hp.max_epochs,
        gradient_clip_val=hp.gradient_clip_val,
        logger=loggers if not args.predict else [],
        callbacks=callbacks,
        log_every_n_steps=0,
    )

    # ========================================================================
    # Execution
    # ========================================================================

    if not args.predict:

        def on_interrupt(_signum, _frame):
            print("\nTraining interrupted. Saving checkpoint and predictions...")
            trainer.save_checkpoint(model_path, weights_only=False)
            trainer.predict(module, dm)
            clean_dir(temp_dir)
            print(f"\nResults saved to {run_dir}")
            with contextlib.suppress(Exception):
                webbrowser.open(str(plot_path))
            sys.exit(0)

        signal.signal(signal.SIGINT, on_interrupt)
        trainer.fit(module, dm)
        trainer.save_checkpoint(model_path, weights_only=False)

    trainer.predict(module, dm)
    clean_dir(temp_dir)

    print(f"\nResults saved to {run_dir}")
    with contextlib.suppress(Exception):
        webbrowser.open(str(plot_path))


if __name__ == "__main__":
    main()
