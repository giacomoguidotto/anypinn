from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys

import torch

from config import CONFIG, hp
from ode import create_context, create_problem


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
            print("\nTraining interrupted. Saving model...")
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
