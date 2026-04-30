# Understanding the Output

After training completes, your project directory contains several new files.
This page explains what each one is and how to use it.

---

## Directory layout after training

```
my-project/
├── ode.py
├── config.py
├── train.py
├── data/
├── logs/                          # Training logs
│   ├── tensorboard/               # TensorBoard event files
│   │   └── <experiment>/<run>/
│   └── csv/                       # CSV-format metric logs
│       └── <experiment>/<run>/
└── models/                        # Saved artifacts
    └── <experiment>/<run>/
        ├── model.ckpt             # Full Lightning checkpoint
        └── predictions.pt         # Predicted fields + recovered parameters
```

The `<experiment>` name comes from `EXPERIMENT_NAME` in `config.py`, and
`<run>` defaults to `v0` (incremented automatically on subsequent runs).

---

## Training logs

### TensorBoard

Launch TensorBoard to visualize training curves:

```bash
uv run tensorboard --logdir logs/tensorboard
```

Key metrics logged every epoch:

| Metric | Meaning |
| ------ | ------- |
| `loss/total` | Weighted sum of all constraint losses |
| `loss/residuals` | ODE/PDE residual on collocation points |
| `loss/ic` | Initial condition error |
| `loss/data` | Fit to observed data |
| `val/<param>_mse` | MSE between recovered and ground-truth parameter |

The `val/*` metrics are only logged when a `ValidationRegistry` is provided
(all built-in templates include one).

### CSV logs

The same metrics are written as CSV files under `logs/csv/`. These are useful
for post-processing or plotting with your own scripts.

---

## Predictions

The `predictions.pt` file is a Python dictionary saved with `torch.save`. Load
it with:

```python
import torch

predictions = torch.load("models/<experiment>/<run>/predictions.pt")
batch, preds, trues = predictions
```

It contains three items:

| Item | Type | Contents |
| ---- | ---- | -------- |
| `batch` | `tuple[Tensor, Tensor]` | `(x_data, y_data)` — the original training data |
| `preds` | `dict[str, Tensor]` | Predicted fields and parameters, keyed by name |
| `trues` | `dict[str, Tensor]` or `None` | Ground-truth values (if validation was configured) |

### Re-running predictions

To reload a trained checkpoint and produce predictions without retraining:

```bash
uv run train.py --predict
```

This skips training entirely and runs the prediction step on the saved
checkpoint.

---

## Checkpoints

The `model.ckpt` file is a standard PyTorch Lightning checkpoint. It contains
the full model state (fields, parameters, optimizer, scheduler) and can be
loaded for:

- **Resuming training** — Lightning's `Trainer` handles this automatically
- **Transfer learning** — load fields from a trained model into a new problem
- **Inspection** — examine recovered parameter values directly

```python
import torch

ckpt = torch.load("models/<experiment>/<run>/model.ckpt")
state_dict = ckpt["state_dict"]

# Recovered scalar parameter value
print(state_dict["problem.params.beta.value"])
```

---

## Next

Now that you understand what the output means, see [Next Steps](next-steps.md)
for where to go from here.
