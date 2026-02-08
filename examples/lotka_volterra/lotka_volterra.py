from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import Tensor

from pinn.catalog.lotka_volterra import (
    ALPHA_KEY,
    BETA_KEY,
    DELTA_KEY,
    GAMMA_KEY,
    X_KEY,
    Y_KEY,
    LotkaVolterraDataModule,
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
TRUE_ALPHA = 0.5
TRUE_BETA = 0.02
TRUE_DELTA = 0.01
TRUE_GAMMA = 0.5

# Initial conditions (populations)
X0 = 40.0
Y0 = 9.0

# Time domain
T_TOTAL = 50

# Population scaling
POP_SCALE = 100.0

# Noise level for synthetic data (std as fraction of signal)
NOISE_FRAC = 0.02


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
            output_activation="softplus",
        ),
        params_config=ScalarConfig(
            init_value=0.05,
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
    # Ground truth for logging/validation: beta is a constant.
    # ========================================================================

    validation: ValidationRegistry = {
        BETA_KEY: lambda x: torch.full_like(x, TRUE_BETA),
    }

    # ============================================================================
    # Training and Prediction Data Definition
    # ============================================================================

    def LV_unscaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
        prey, predator = y
        b = args[BETA_KEY]
        alpha = args[ALPHA_KEY]
        delta = args[DELTA_KEY]
        gamma = args[GAMMA_KEY]
        dx = alpha(x) * prey - b(x) * prey * predator
        dy = delta(x) * prey * predator - gamma(x) * predator
        return torch.stack([dx, dy])

    gen_props = ODEProperties(
        ode=LV_unscaled,
        y0=torch.tensor([X0, Y0]),
        args={
            ALPHA_KEY: Argument(TRUE_ALPHA),
            BETA_KEY: Argument(TRUE_BETA),
            DELTA_KEY: Argument(TRUE_DELTA),
            GAMMA_KEY: Argument(TRUE_GAMMA),
        },
    )

    dm = LotkaVolterraDataModule(
        hp=hp,
        gen_props=gen_props,
        noise_frac=NOISE_FRAC,
        validation=validation,
        callbacks=[DataScaling(y_scale=[1 / POP_SCALE, 1 / POP_SCALE])],
    )

    # ========================================================================
    # Problem Definition
    # The scaled Lotka-Volterra ODE: derivatives multiplied by T to account
    # for time normalization to [0, 1]. Populations are scaled by POP_SCALE.
    # ========================================================================

    def LV_scaled(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
        prey, predator = y
        b = args[BETA_KEY]
        alpha = args[ALPHA_KEY]
        delta = args[DELTA_KEY]
        gamma = args[GAMMA_KEY]

        dx = alpha(x) * prey - b(x) * prey * predator * POP_SCALE
        dy = delta(x) * prey * predator * POP_SCALE - gamma(x) * predator

        dx = dx * T
        dy = dy * T
        return torch.stack([dx, dy])

    props = ODEProperties(
        ode=LV_scaled,
        y0=torch.tensor([X0, Y0]) / POP_SCALE,
        args={
            ALPHA_KEY: Argument(TRUE_ALPHA),
            DELTA_KEY: Argument(TRUE_DELTA),
            GAMMA_KEY: Argument(TRUE_GAMMA),
        },
    )

    fields = FieldsRegistry(
        {
            X_KEY: Field(config=hp.fields_config),
            Y_KEY: Field(config=hp.fields_config),
        }
    )
    params = ParamsRegistry(
        {
            BETA_KEY: Parameter(config=hp.params_config),
        }
    )

    def predict_data(x_data: Tensor, fields: FieldsRegistry, _params: ParamsRegistry) -> Tensor:
        x_pred = fields[X_KEY](x_data)
        y_pred = fields[Y_KEY](x_data)
        return torch.stack([x_pred, y_pred])

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
    t_data, y_data = batch

    # Unscale predictions back to original population scale
    x_pred = POP_SCALE * preds[X_KEY]
    y_pred = POP_SCALE * preds[Y_KEY]

    # Unscale observed data
    x_obs = POP_SCALE * y_data[:, 0]
    y_obs = POP_SCALE * y_data[:, 1]

    beta_pred = preds[BETA_KEY]
    beta_true = trues[BETA_KEY] if trues else None

    # plot
    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: prey and predator curves + observed data
    ax = axes[0]
    sns.lineplot(x=t_data, y=x_pred, label="Prey $x_{pred}$", ax=ax, color="C0")
    sns.lineplot(x=t_data, y=y_pred, label="Predator $y_{pred}$", ax=ax, color="C3")
    sns.scatterplot(x=t_data, y=x_obs, label="Prey $x_{obs}$", ax=ax, color="C0", s=10, alpha=0.4)
    sns.scatterplot(
        x=t_data, y=y_obs, label="Predator $y_{obs}$", ax=ax, color="C3", s=10, alpha=0.4
    )
    ax.set_title("Lotka-Volterra Predictions")
    ax.set_xlabel("Time (scaled)")
    ax.set_ylabel("Population")
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
            "x_obs": x_obs,
            "y_obs": y_obs,
            "x_pred": x_pred,
            "y_pred": y_pred,
            "beta_pred": beta_pred,
            "beta_true": beta_true,
        }
    )

    df.to_csv(predictions_dir / "predictions.csv", index=False, float_format="%.6e")


# ============================================================================
# Main
# ============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lotka-Volterra Inverse Example")
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Load saved model and run prediction. Does not train the model.",
    )
    args = parser.parse_args()

    experiment_name = "lotka-volterra"
    run_name = "v0-synthetic-alt"

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
