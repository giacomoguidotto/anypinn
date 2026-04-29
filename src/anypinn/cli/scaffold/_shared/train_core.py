from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys

from config import EXPERIMENT_NAME, RUN_NAME, hp
from ode import create_data_module, create_problem
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="__EXPERIMENT_NAME__")
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Load saved model and run prediction only.",
    )
    args = parser.parse_args()

    # ========================================================================
    # Setup
    # ========================================================================

    models_dir = Path("./models") / EXPERIMENT_NAME / RUN_NAME
    models_dir.mkdir(exist_ok=True, parents=True)
    model_path = models_dir / "model.pt"

    problem = create_problem(hp)
    dm = create_data_module(hp)
    dm.setup("fit")
    problem.inject_context(dm.context)

    if args.predict:
        problem.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        optimizer = torch.optim.Adam(problem.parameters(), lr=hp.lr)

        def on_interrupt(_signum, _frame):
            print("\nTraining interrupted. Saving model...")
            torch.save(problem.state_dict(), model_path)
            sys.exit(0)

        signal.signal(signal.SIGINT, on_interrupt)

        # Change these parameters to suit your problem
        train_dl = dm.train_dataloader()
        for epoch in range(hp.max_epochs):
            epoch_loss = 0.0
            for batch in train_dl:
                optimizer.zero_grad()
                loss = problem.training_loss(batch, log=None)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{hp.max_epochs} — loss: {epoch_loss:.4e}")

        torch.save(problem.state_dict(), model_path)

    print("Done.")


if __name__ == "__main__":
    main()
