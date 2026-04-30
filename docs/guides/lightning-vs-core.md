# Lightning vs Core Training

AnyPINN's `Problem` is a plain `nn.Module`. You can train it with PyTorch
Lightning or with a raw PyTorch loop. This guide helps you choose and shows
both approaches.

---

## When to use which

| | Lightning | Core |
| --- | --- | --- |
| **Best for** | Standard training with logging and callbacks | Custom training procedures |
| **Setup effort** | Minimal: `anypinn create --lightning` generates everything | You write the loop |
| **Logging** | TensorBoard + CSV out of the box | You call `log()` yourself |
| **Checkpointing** | Automatic | You call `torch.save()` |
| **Early stopping** | Built-in callback | You implement the logic |
| **Multi-GPU** | Lightning handles it | You set up DDP |
| **Flexibility** | Callback-based customization | Full control |

**Default recommendation:** Start with Lightning. Drop to core only when you
need something Lightning doesn't support.

---

## Lightning training

Scaffold with `--lightning` (the default):

```bash
anypinn create my-project --template sir --data synthetic --lightning
```

The generated `train.py` wires up:

```python
from anypinn.lightning import PINNModule

module = PINNModule(problem=problem, hp=hp)

trainer = pl.Trainer(
    max_epochs=hp.max_epochs,
    gradient_clip_val=hp.gradient_clip_val,
    callbacks=[...],
    logger=[...],
)

trainer.fit(module, datamodule=data_module)
```

`PINNModule` handles:

- Forwarding `training_loss()` to the `Problem`
- Configuring the optimizer and scheduler from `hp`
- Injecting `InferredContext` (domain bounds, data statistics) at fit start
- Formatting predictions for the `PredictionsWriter` callback

---

## Core training

Scaffold with `--no-lightning`:

```bash
anypinn create my-project --template sir --data synthetic --no-lightning
```

The generated `train.py` is a standard PyTorch loop:

```python
optimizer = torch.optim.Adam(problem.parameters(), lr=hp.lr)

for epoch in range(hp.max_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = problem.training_loss(batch, log=print)
        loss.backward()
        optimizer.step()
```

The `Problem` is an `nn.Module`, so everything works exactly as you'd expect
from PyTorch.

---

## Mixing approaches

You can use core-level primitives inside a Lightning training setup:

```python
class CustomModule(pl.LightningModule):
    def __init__(self, problem, hp):
        super().__init__()
        self.problem = problem
        self.hp = hp

    def training_step(self, batch, batch_idx):
        loss = self.problem.training_loss(batch, log=self.log)
        # Add your custom logic here
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hp.lr)
```

Or use the built-in `PINNModule` and add behavior through Lightning callbacks:

```python
class MyCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Custom logic after each epoch
        ...

trainer = pl.Trainer(callbacks=[MyCallback()])
```

---

## Key differences in practice

### Logging

**Lightning:** Metrics are logged automatically. Access them in TensorBoard.

**Core:** Pass a logging function to `training_loss`:

```python
def my_logger(key: str, value: Tensor, progress_bar: bool = False):
    print(f"{key}: {value.item():.6f}")

loss = problem.training_loss(batch, log=my_logger)
```

### Context injection

**Lightning:** `PINNModule.on_fit_start()` automatically injects domain bounds
and data statistics into the problem.

**Core:** You must set context manually:

```python
from anypinn.core import InferredContext

context = InferredContext.from_data(x_data, y_data)
problem.set_context(context)
```

### Predictions

**Lightning:** Use `trainer.predict()` with the `PredictionsWriter` callback.

**Core:** Call the fields and parameters directly:

```python
with torch.no_grad():
    y_pred = {key: field(x_test) for key, field in problem.fields.items()}
    p_pred = {key: param(x_test) for key, param in problem.params.items()}
```
