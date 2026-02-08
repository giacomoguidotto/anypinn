from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import shutil
from typing import cast

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import Tensor

from pinn.catalog.damped_oscillator import (
    OMEGA_KEY,
    V_KEY,
    X_KEY,
    ZETA_KEY,
    DampedOscillatorDataModule,
)
from pinn.core import (
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
    ScalarConfig,
    SchedulerConfig,
    ValidationRegistry,
)
from pinn.lightning import PINNModule, SMMAStopping
from pinn.lightning.callbacks import DataScaling, FormattedProgressBar, Metric, PredictionsWriter
from pinn.problems import ODEHyperparameters, ODEInverseProblem, ODEProperties

# ============================================================================
# Constants
# ============================================================================

# True parameter values
TRUE_ZETA = 0.15
TRUE_OMEGA0 = 2 * math.pi

# Initial conditions
X0 = 1.0
V0 = 0.0

# Time domain (seconds)
T_TOTAL = 5

# Noise level for synthetic data
NOISE_STD = 0.02


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
    # Time scaling constant
    T = T_TOTAL

    # ========================================================================
    # Hyperparameters
    # ========================================================================

    hp = ODEHyperparameters(
        lr=1e-3,
        training_data=GenerationConfig(
            batch_size=100,
            data_ratio=2,
            collocations=6000,
            x=torch.linspace(start=0, end=T, steps=200),
            noise_level=0,
            args_to_train={},
        ),
        fields_config=MLPConfig(
            in_dim=1,
            out_dim=1,
            hidden_layers=[64, 128, 128, 64],
            activation="tanh",
            output_activation=None,
        ),
        params_config=ScalarConfig(
            init_value=0.3,
        ),
        scheduler=SchedulerConfig(
            mode="min",
            factor=0.5,
            patience=55,
            threshold=5e-3,
            min_lr=1e-6,
        ),
        pde_weight=1e-4,
        ic_weight=15,
        data_weight=5,
    )

    # ========================================================================
    # Validation Configuration
    # Ground truth for logging/validation: zeta is a constant.
    # ========================================================================

    validation: ValidationRegistry = {
        ZETA_KEY: lambda x: torch.full_like(x, TRUE_ZETA),
    }

    # ============================================================================
    # Training and Prediction Data Definition
    # ============================================================================

    def oscillator_unscaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
        pos, vel = y
        z = args[ZETA_KEY]
        omega0 = args[OMEGA_KEY]
        dx = vel
        dv = -2 * z(x) * omega0(x) * vel - omega0(x) ** 2 * pos
        return torch.stack([dx, dv])

    gen_props = ODEProperties(
        ode=oscillator_unscaled,
        y0=torch.tensor([X0, V0]),
        args={
            ZETA_KEY: Argument(TRUE_ZETA),
            OMEGA_KEY: Argument(TRUE_OMEGA0),
        },
    )

    dm = DampedOscillatorDataModule(
        hp=hp,
        gen_props=gen_props,
        noise_std=NOISE_STD,
        validation=validation,
        callbacks=[DataScaling(y_scale=1.0)],
    )

    # ========================================================================
    # Problem Definition
    # The scaled damped oscillator ODE: derivatives multiplied by T to
    # account for time normalization to [0, 1].
    #   dx/dt = v
    #   dv/dt = -2*zeta*omega0*v - omega0^2*x
    # ========================================================================

    def oscillator_scaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
        pos, vel = y
        z = args[ZETA_KEY]
        omega0 = args[OMEGA_KEY]

        dx = vel
        dv = -2 * z(x) * omega0(x) * vel - omega0(x) ** 2 * pos

        dx = dx * T
        dv = dv * T
        return torch.stack([dx, dv])

    props = ODEProperties(
        ode=oscillator_scaled,
        y0=torch.tensor([X0, V0]),
        args={
            OMEGA_KEY: Argument(TRUE_OMEGA0),
        },
    )

    fields = FieldsRegistry(
        {
            X_KEY: Field(config=hp.fields_config),
            V_KEY: Field(config=hp.fields_config),
        }
    )
    params = ParamsRegistry(
        {
            ZETA_KEY: Parameter(config=hp.params_config),
        }
    )

    def predict_data(x_data: Tensor, fields: FieldsRegistry, _params: ParamsRegistry) -> Tensor:
        x_pred = fields[X_KEY](x_data)
        return cast(Tensor, x_pred)

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
        try:
            trainer.fit(module, dm)
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving checkpoint and predictions...")
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
    t_data, x_data = batch

    x_pred = preds[X_KEY]
    v_pred = preds[V_KEY]

    zeta_pred = preds[ZETA_KEY]
    zeta_true = trues[ZETA_KEY] if trues else None

    # plot
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: position + data, velocity predicted
    ax = axes[0]
    sns.lineplot(x=t_data, y=x_pred, label="$x_{pred}$ (position)", ax=ax, color="C0")
    sns.lineplot(x=t_data, y=v_pred, label="$v_{pred}$ (velocity)", ax=ax, color="C2")
    sns.scatterplot(x=t_data, y=x_data, label="$x_{observed}$", ax=ax, color="C1", s=10, alpha=0.5)
    ax.set_title("Damped Oscillator Predictions")
    ax.set_xlabel("Time (scaled)")
    ax.set_ylabel("Amplitude")
    ax.legend()

    # Subplot 2: zeta predicted vs true
    ax = axes[1]
    if zeta_true is not None:
        sns.lineplot(x=t_data, y=zeta_true, label=r"$\zeta_{true}$", ax=ax, color="C0")
    sns.lineplot(
        x=t_data,
        y=zeta_pred,
        label=r"$\zeta_{pred}$",
        linestyle="--" if zeta_true is not None else "-",
        ax=ax,
        color="C3" if zeta_true is not None else "C0",
    )
    ax.set_title(r"$\zeta$ (Damping Ratio) Prediction")
    ax.set_xlabel("Time (scaled)")
    ax.set_ylabel(r"$\zeta$")
    ax.legend()

    plt.tight_layout()

    fig.savefig(predictions_dir / "predictions.png", dpi=300)
    plt.close(fig)

    # save
    df = pd.DataFrame(
        {
            "t": t_data,
            "x_observed": x_data,
            "x_pred": x_pred,
            "v_pred": v_pred,
            "zeta_pred": zeta_pred,
            "zeta_true": zeta_true,
        }
    )

    df.to_csv(predictions_dir / "predictions.csv", index=False, float_format="%.6e")


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Damped Oscillator Inverse Example")
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Load saved model and run prediction. Does not train the model.",
    )
    args = parser.parse_args()

    experiment_name = "damped-oscillator"
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
