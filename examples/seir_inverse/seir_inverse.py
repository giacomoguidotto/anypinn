from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import signal
import sys
from typing import cast

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import Tensor

from anypinn.catalog.seir import (
    BETA_KEY,
    E_KEY,
    GAMMA_KEY,
    I_KEY,
    S_KEY,
    SIGMA_KEY,
    SEIRDataModule,
)
from anypinn.core import (
    LOSS_KEY,
    ArgsRegistry,
    Argument,
    Field,
    FieldsRegistry,
    GenerationConfig,
    MLPConfig,
    Parameter,
    ParamsRegistry,
    Predictions,
    SchedulerConfig,
    ValidationRegistry,
)
from anypinn.lightning import PINNModule, SMMAStopping
from anypinn.lightning.callbacks import (
    DataScaling,
    FormattedProgressBar,
    Metric,
    PredictionsWriter,
)
from anypinn.problems import ODEHyperparameters, ODEInverseProblem, ODEProperties

# ============================================================================
# Constants
# ============================================================================

# True parameter values
TRUE_BETA = 0.5
TRUE_SIGMA = 1 / 5.2
TRUE_GAMMA = 1 / 10

# Initial conditions (fractions)
S0 = 0.99
E0 = 0.01
I0 = 0.001

# Time domain
T_DAYS = 160

# Noise level for synthetic data
NOISE_STD = 0.0005


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class RunConfig:
    max_epochs: int
    gradient_clip_val: float
    predict: bool

    run_name: str
    tensorboard_dir: Path
    csv_dir: Path
    model_path: Path
    predictions_dir: Path
    checkpoint_dir: Path
    experiment_name: str


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


def main(config: RunConfig) -> None:
    # Time scaling constant (data spans T_DAYS days, scaled to [0,1])
    T = T_DAYS

    # ========================================================================
    # Hyperparameters
    # ========================================================================

    hp = ODEHyperparameters(
        lr=5e-4,
        training_data=GenerationConfig(
            batch_size=100,
            data_ratio=2,
            collocations=6000,
            x=torch.linspace(start=0, end=T, steps=T + 1),
            noise_level=0,
            args_to_train={},
        ),
        fields_config=MLPConfig(
            in_dim=1,
            out_dim=1,
            hidden_layers=[64, 128, 128, 64],
            activation="tanh",
            output_activation="softplus",
        ),
        params_config=MLPConfig(
            in_dim=1,
            out_dim=1,
            hidden_layers=[64, 128, 128, 64],
            activation="tanh",
            output_activation="softplus",
        ),
        scheduler=SchedulerConfig(
            mode="min",
            factor=0.5,
            patience=55,
            threshold=5e-3,
            min_lr=1e-6,
        ),
        pde_weight=1,
        ic_weight=1,
        data_weight=1,
    )

    # ========================================================================
    # Validation Configuration
    # Ground truth for logging/validation: beta is a constant.
    # ========================================================================

    validation: ValidationRegistry = {
        BETA_KEY: lambda x: torch.full_like(x, TRUE_BETA),
    }

    # ============================================================================
    # Training and Prediction Data Definition
    # ============================================================================

    def SEIR_unscaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
        S, E, I = y
        b = args[BETA_KEY]
        sigma = args[SIGMA_KEY]
        gamma = args[GAMMA_KEY]
        dS = -b(x) * S * I
        dE = b(x) * S * I - sigma(x) * E
        dI = sigma(x) * E - gamma(x) * I
        return torch.stack([dS, dE, dI])

    gen_props = ODEProperties(
        ode=SEIR_unscaled,
        y0=torch.tensor([S0, E0, I0]),
        args={
            BETA_KEY: Argument(TRUE_BETA),
            SIGMA_KEY: Argument(TRUE_SIGMA),
            GAMMA_KEY: Argument(TRUE_GAMMA),
        },
    )

    dm = SEIRDataModule(
        hp=hp,
        gen_props=gen_props,
        noise_std=NOISE_STD,
        validation=validation,
        callbacks=[DataScaling(y_scale=1.0)],
    )

    # ========================================================================
    # Problem Definition
    # The scaled SEIR ODE: derivatives multiplied by T to account for
    # time normalization to [0, 1].
    # ========================================================================

    def SEIR_scaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
        S, E, I = y
        b = args[BETA_KEY]
        sigma = args[SIGMA_KEY]
        gamma = args[GAMMA_KEY]

        dS = -b(x) * S * I
        dE = b(x) * S * I - sigma(x) * E
        dI = sigma(x) * E - gamma(x) * I

        dS = dS * T
        dE = dE * T
        dI = dI * T
        return torch.stack([dS, dE, dI])

    props = ODEProperties(
        ode=SEIR_scaled,
        y0=torch.tensor([S0, E0, I0]),
        args={
            SIGMA_KEY: Argument(TRUE_SIGMA),
            GAMMA_KEY: Argument(TRUE_GAMMA),
        },
    )

    fields = FieldsRegistry(
        {
            S_KEY: Field(config=hp.fields_config),
            E_KEY: Field(config=hp.fields_config),
            I_KEY: Field(config=hp.fields_config),
        }
    )
    params = ParamsRegistry(
        {
            BETA_KEY: Parameter(config=hp.params_config),
        }
    )

    def predict_data(x_data: Tensor, fields: FieldsRegistry, _params: ParamsRegistry) -> Tensor:
        I = fields[I_KEY]
        I_pred = I(x_data)
        return cast(Tensor, I_pred)

    problem = ODEInverseProblem(
        props=props,
        hp=hp,
        fields=fields,
        params=params,
        predict_data=predict_data,
    )

    # ============================================================================
    # Training Modules Definition
    # ============================================================================

    if config.predict:
        module = PINNModule.load_from_checkpoint(
            config.model_path,
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
            dirpath=config.checkpoint_dir,
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
            predictions_path=config.predictions_dir / "predictions.pt",
            on_prediction=lambda _, __, predictions_list, ___: plot_and_save(
                predictions_list[0], config.predictions_dir
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
            save_dir=config.tensorboard_dir,
            name=config.experiment_name,
            version=config.run_name,
        ),
        CSVLogger(
            save_dir=config.csv_dir,
            name=config.experiment_name,
            version=config.run_name,
        ),
    ]

    trainer = Trainer(
        max_epochs=config.max_epochs,
        gradient_clip_val=config.gradient_clip_val,
        logger=loggers if not config.predict else [],
        callbacks=callbacks,
        log_every_n_steps=0,
    )

    # ============================================================================
    # Execution
    # ============================================================================

    if not config.predict:

        def on_interrupt(_signum, _frame):
            print("\nTraining interrupted. Saving checkpoint and predictions...")
            trainer.save_checkpoint(config.model_path, weights_only=False)
            trainer.predict(module, dm)
            clean_dir(config.checkpoint_dir)
            sys.exit(0)

        signal.signal(signal.SIGINT, on_interrupt)
        trainer.fit(module, dm)
        trainer.save_checkpoint(config.model_path, weights_only=False)

    trainer.predict(module, dm)

    clean_dir(config.checkpoint_dir)


# ============================================================================
# Plotting and Saving
# ============================================================================


def plot_and_save(
    predictions: Predictions,
    predictions_dir: Path,
) -> None:
    batch, preds, trues = predictions
    t_data, I_data = batch

    S_pred = preds[S_KEY]
    E_pred = preds[E_KEY]
    I_pred = preds[I_KEY]

    beta_pred = preds[BETA_KEY]
    beta_true = trues[BETA_KEY] if trues else None

    # plot
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: S, E, I curves + observed I data
    ax = axes[0]
    sns.lineplot(x=t_data, y=S_pred, label="$S_{pred}$", ax=ax, color="C0")
    sns.lineplot(x=t_data, y=E_pred, label="$E_{pred}$", ax=ax, color="C2")
    sns.lineplot(x=t_data, y=I_pred, label="$I_{pred}$", ax=ax, color="C3")
    sns.scatterplot(x=t_data, y=I_data, label="$I_{observed}$", ax=ax, color="C1", s=10, alpha=0.5)
    ax.set_title("SEIR Model Predictions")
    ax.set_xlabel("Time (scaled)")
    ax.set_ylabel("Fraction")
    ax.legend()

    # Subplot 2: beta predicted vs true
    ax = axes[1]
    if beta_true is not None:
        sns.lineplot(x=t_data, y=beta_true, label=r"$\beta_{true}$", ax=ax, color="C0")
    sns.lineplot(
        x=t_data,
        y=beta_pred,
        label=r"$\beta_{pred}$",
        linestyle="--" if beta_true is not None else "-",
        ax=ax,
        color="C3" if beta_true is not None else "C0",
    )
    ax.set_title(r"$\beta$ Parameter Prediction")
    ax.set_xlabel("Time (scaled)")
    ax.set_ylabel(r"$\beta$")
    ax.legend()

    plt.tight_layout()

    fig.savefig(predictions_dir / "predictions.png", dpi=300)
    plt.close(fig)

    # save
    df = pd.DataFrame(
        {
            "t": t_data,
            "I_observed": I_data,
            "S_pred": S_pred,
            "E_pred": E_pred,
            "I_pred": I_pred,
            "beta_pred": beta_pred,
            "beta_true": beta_true,
        }
    )

    df.to_csv(predictions_dir / "predictions.csv", index=False, float_format="%.6e")


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEIR Inverse Example")
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Load saved model and run prediction. Does not train the model.",
    )
    args = parser.parse_args()

    experiment_name = "seir-inverse"
    run_name = "v0-synthetic"

    log_dir = Path("./logs")
    tensorboard_dir = log_dir / "tensorboard"
    csv_dir = log_dir / "csv"

    models_dir = Path("./models") / experiment_name / run_name
    model_path = models_dir / "model.ckpt"
    predictions_dir = models_dir

    temp_dir = Path("./temp")

    create_dir(log_dir)
    create_dir(models_dir)
    create_dir(predictions_dir)
    create_dir(temp_dir)

    clean_dir(temp_dir)
    if not args.predict:
        clean_dir(csv_dir / experiment_name / run_name)
        clean_dir(tensorboard_dir / experiment_name / run_name)

    config = RunConfig(
        max_epochs=2000,
        gradient_clip_val=0.1,
        predict=args.predict,
        run_name=run_name,
        tensorboard_dir=tensorboard_dir,
        csv_dir=csv_dir,
        model_path=model_path,
        predictions_dir=predictions_dir,
        checkpoint_dir=temp_dir,
        experiment_name=experiment_name,
    )

    main(config)
