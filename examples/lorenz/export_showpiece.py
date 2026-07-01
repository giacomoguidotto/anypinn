"""Export per-epoch Lorenz snapshots for the guidotto.dev showpiece.

Showpiece utility. This file is not part of the AnyPINN library API. It re-runs
the existing Lorenz inverse example (reusing the unmodified `config.py` / `ode.py`
factories) and, at log-spaced epochs, snapshots:

  - the three field MLPs' weights (x, y, z) -> drives the live in-browser
    forward pass that traces the attractor as it converges;
  - the recovered scalars sigma, rho, beta -> drives the "converging knobs" HUD;
  - the training loss at that epoch -> drives the HUD loss readout.

It writes a faithful float32 binary + JSON manifest the website can later quantize
and compress. Nothing in the real library is touched.

Run it the same way you run the example (from this directory), e.g.:

    uv run python export_showpiece.py
    # optional: --out /some/dir   --snapshots 40

Default output: /Users/giacomo/dev/life/guidotto.dev/.showpiece/lorenz/ (gitignored).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from config import hp
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback
import numpy as np
import ode
from ode import (
    SCALE,
    T_TOTAL,
    TRUE_BETA,
    TRUE_RHO,
    TRUE_SIGMA,
    X0,
    Y0,
    Z0,
    create_data_module,
    create_problem,
)
import torch
import torch.nn as nn

from anypinn.core import LOSS_KEY
from anypinn.lightning import PINNModule

FIELD_ORDER = ("x", "y", "z")
PARAM_ORDER = ("sigma", "rho", "beta")

DEFAULT_OUT = Path("/Users/giacomo/dev/life/guidotto.dev/.showpiece/lorenz")


def make_schedule(max_epochs: int, n: int) -> set[int]:
    """Log-spaced "epochs completed" targets in [1, max_epochs], final always in."""
    pts = np.unique(np.round(np.geomspace(1, max_epochs, num=n)).astype(int))
    targets = {int(p) for p in pts if 1 <= p <= max_epochs}
    targets.add(int(max_epochs))
    return targets


class SnapshotExporter(Callback):
    """Captures field weights + recovered params + loss at scheduled epochs."""

    def __init__(self, targets: set[int]) -> None:
        self._targets = targets
        self._epoch = 0
        self.snapshots: list[dict] = []

    @staticmethod
    def _field_weights(problem) -> np.ndarray:
        """Flatten every Linear (weight then bias) across x, y, z fields, in order."""
        chunks: list[np.ndarray] = []
        for fk in FIELD_ORDER:
            for layer in problem.fields[fk].net:
                if isinstance(layer, nn.Linear):
                    chunks.append(layer.weight.detach().cpu().float().reshape(-1).numpy().copy())
                    chunks.append(layer.bias.detach().cpu().float().reshape(-1).numpy().copy())
        return np.concatenate(chunks).astype("<f4")

    @staticmethod
    def _params(problem) -> dict[str, float]:
        return {pk: float(problem.params[pk].value.detach().cpu()) for pk in PARAM_ORDER}

    def _grab(self, pl_module, epoch: int, loss: float | None) -> None:
        problem = pl_module.problem
        self.snapshots.append(
            {
                "epoch": epoch,
                "loss": loss,
                "params": self._params(problem),
                "weights": self._field_weights(problem),
            }
        )

    def on_fit_start(self, trainer, pl_module) -> None:
        # Pristine initialization: the "from chaos" start frame (knobs all at init).
        self._grab(pl_module, epoch=0, loss=None)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self._epoch += 1
        if self._epoch in self._targets:
            metric = trainer.callback_metrics.get(LOSS_KEY)
            loss = float(metric) if metric is not None else None
            self._grab(pl_module, epoch=self._epoch, loss=loss)

    def on_fit_end(self, trainer, pl_module) -> None:
        # Guarantee the final converged state is present even if scheduling missed it.
        if not self.snapshots or self.snapshots[-1]["epoch"] != self._epoch:
            metric = trainer.callback_metrics.get(LOSS_KEY)
            loss = float(metric) if metric is not None else None
            self._grab(pl_module, epoch=self._epoch, loss=loss)


def capture_observations(trainer: Trainer, module: PINNModule, dm) -> dict:
    """The 300 noisy data points the net is handed (the floating 'motes')."""
    results = trainer.predict(module, dm)
    (tau, y_scaled), _preds, _trues = results[0]
    tau = tau.detach().cpu().float().numpy()  # input grid in [0, 1]
    y_scaled = y_scaled.detach().cpu().float().numpy()  # (N, 3) scaled state
    physical = SCALE * y_scaled
    return {
        "count": int(tau.shape[0]),
        "tau": tau.tolist(),  # normalized network-input time, [0, 1]
        "t": (T_TOTAL * tau).tolist(),  # physical time, [0, T_TOTAL]
        "x": physical[:, 0].tolist(),
        "y": physical[:, 1].tolist(),
        "z": physical[:, 2].tolist(),
    }


def write_artifacts(out_dir: Path, exporter: SnapshotExporter, observations: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_path = out_dir / "weights.bin"
    snapshot_meta: list[dict] = []
    offset = 0
    with weights_path.open("wb") as fh:
        for snap in exporter.snapshots:
            raw = snap["weights"].tobytes()
            fh.write(raw)
            snapshot_meta.append(
                {
                    "epoch": snap["epoch"],
                    "loss": snap["loss"],
                    "sigma": snap["params"]["sigma"],
                    "rho": snap["params"]["rho"],
                    "beta": snap["params"]["beta"],
                    "byteOffset": offset,
                    "byteLength": len(raw),
                    "floatCount": snap["weights"].size,
                }
            )
            offset += len(raw)

    # Per-field linear-layer shapes (PyTorch weight is [out, in], row-major).
    dims = [1, *hp.fields_config.hidden_layers, 1]
    layers = [{"in": dims[i], "out": dims[i + 1]} for i in range(len(dims) - 1)]

    manifest = {
        "system": "lorenz",
        "note": (
            "Each snapshot = 3 field MLPs (x,y,z). Forward: tau in [0,1] -> net "
            "-> *SCALE = physical state. Physical time t = T_TOTAL * tau."
        ),
        "dtype": "float32",
        "byteOrder": "little",
        "weightsFile": "weights.bin",
        "fieldOrder": list(FIELD_ORDER),
        "paramOrder": list(PARAM_ORDER),
        "activation": hp.fields_config.activation,
        "layers": layers,
        "weightLayout": (
            "Per snapshot: for each field in fieldOrder, for each linear layer in order, "
            "weight[out*in] row-major (out-major) immediately followed by bias[out]. "
            "Fields concatenated. y = tanh between layers, none after the last."
        ),
        "scale": SCALE,
        "tTotal": T_TOTAL,
        "noiseStd": float(ode.NOISE_STD),
        "inputDomain": [0.0, 1.0],
        "y0": [X0, Y0, Z0],
        "trueParams": {"sigma": TRUE_SIGMA, "rho": TRUE_RHO, "beta": TRUE_BETA},
        "initParam": float(hp.params_config.init_value),
        "maxEpochs": int(hp.max_epochs),
        "snapshotCount": len(snapshot_meta),
        "snapshots": snapshot_meta,
    }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "observations.json").write_text(json.dumps(observations, indent=2))

    total_mb = weights_path.stat().st_size / 1e6
    print(f"\nWrote {len(snapshot_meta)} snapshots -> {out_dir}")
    print(f"  weights.bin      {total_mb:.2f} MB")
    print(f"  manifest.json    {len(snapshot_meta)} snapshots, epochs "
          f"{snapshot_meta[0]['epoch']}..{snapshot_meta[-1]['epoch']}")
    print(f"  observations.json {observations['count']} points")
    final = snapshot_meta[-1]
    print(f"  final recovered: sigma={final['sigma']:.4f} rho={final['rho']:.4f} "
          f"beta={final['beta']:.4f}  (true 10 / 28 / 2.6667)")


def verify_roundtrip(out_dir: Path, module: PINNModule) -> None:
    """Reconstruct the final snapshot from disk with NumPy per the manifest recipe and
    confirm it reproduces the live model's field outputs. This is the JS-port contract:
    if this matches, a faithful JS implementation of the documented layout is correct."""
    manifest = json.loads((out_dir / "manifest.json").read_text())
    layers = manifest["layers"]
    final = manifest["snapshots"][-1]
    buf = np.fromfile(
        out_dir / "weights.bin",
        dtype="<f4",
        count=final["floatCount"],
        offset=final["byteOffset"],
    )

    tau = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    x_in = tau.reshape(-1, 1)
    o = 0
    max_err = 0.0
    for fk in manifest["fieldOrder"]:
        a = x_in
        for i, lyr in enumerate(layers):
            win, wout = lyr["in"], lyr["out"]
            w = buf[o : o + wout * win].reshape(wout, win)
            o += wout * win
            b = buf[o : o + wout]
            o += wout
            a = a @ w.T + b
            if i < len(layers) - 1:
                a = np.tanh(a)
        with torch.no_grad():
            ref = module.problem.fields[fk](torch.from_numpy(x_in)).cpu().numpy()
        max_err = max(max_err, float(np.abs(a - ref).max()))

    status = "OK" if max_err < 1e-4 else "MISMATCH"
    print(f"  roundtrip verify: max|numpy - model| = {max_err:.2e}  [{status}]")
    if max_err >= 1e-4:
        raise SystemExit("Round-trip verification FAILED — JS-port layout would be wrong.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Lorenz showpiece snapshots.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(os.environ.get("SHOWPIECE_OUT", DEFAULT_OUT)),
    )
    parser.add_argument(
        "--snapshots",
        type=int,
        default=40,
        help="approx. number of log-spaced snapshots",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="override hp.max_epochs (for smoke tests)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--noise",
        type=float,
        default=None,
        help="override ode.NOISE_STD (default 0.5)",
    )
    args = parser.parse_args()

    if args.noise is not None:
        ode.NOISE_STD = args.noise

    seed_everything(args.seed, workers=True)
    print(f"  data noise_std = {ode.NOISE_STD}  seed = {args.seed}")

    dm = create_data_module(hp)
    problem = create_problem(hp)
    module = PINNModule(problem=problem, hp=hp)

    max_epochs = args.max_epochs if args.max_epochs is not None else hp.max_epochs
    targets = make_schedule(max_epochs, args.snapshots)
    exporter = SnapshotExporter(targets)

    trainer = Trainer(
        max_epochs=max_epochs,
        gradient_clip_val=hp.gradient_clip_val,
        logger=False,
        enable_checkpointing=False,
        log_every_n_steps=0,
        callbacks=[exporter],
    )

    trainer.fit(module, dm)
    observations = capture_observations(trainer, module, dm)
    write_artifacts(args.out, exporter, observations)
    verify_roundtrip(args.out, module)


if __name__ == "__main__":
    main()
