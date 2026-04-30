# First Project

This walkthrough creates and trains a complete Physics-Informed Neural Network
that recovers the transmission rate of an epidemic from synthetic observations.
It takes about 5 minutes.

---

## Create the project

The `anypinn create` command scaffolds a complete, runnable project:

```bash
anypinn create my-project
```

The CLI walks through three choices interactively:

### 1. Choose a template

The first prompt selects a physics model. There are 16 built-in templates
spanning ODE and PDE problems:

| Category | Templates |
| -------- | --------- |
| Epidemiology | SIR, SEIR |
| Oscillators | Damped Oscillator, Van der Pol |
| Ecology | Lotka-Volterra |
| Chaotic systems | Lorenz |
| Neuroscience | FitzHugh-Nagumo |
| PDEs | Gray-Scott 2D, Poisson 2D, Heat 1D, Burgers 1D, Wave 1D, Inverse Diffusivity, Allen-Cahn |
| Custom | Custom ODE (stub), Blank (empty) |

Each template ships a complete mathematical definition with ODE/PDE callables,
initial conditions, known parameters, and validation ground truths. Run
`anypinn create --list-templates` to see all options with descriptions.

### 2. Select a data source

Choose between **synthetic** or **CSV**:

- **Synthetic** generates training data by numerically integrating the ODE with
  [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq). This is the right
  starting point for learning the library, since no external data is needed.
- **CSV** loads observations from a file in the `data/` directory. Use this
  when you have real experimental measurements.

### 3. Include Lightning wrapper

Choose whether the generated training script uses
[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) or a plain
PyTorch loop:

- **Yes** (default) generates a full Lightning `Trainer` setup with TensorBoard
  logging, CSV logging, model checkpointing, learning rate monitoring, early
  stopping, and a `--predict` flag for inference.
- **No** generates a minimal loop with `torch.optim.Adam` and manual
  `loss.backward()` / `optimizer.step()`. You own the training loop entirely.

### Non-interactive mode

All three choices can be passed as flags to skip the prompts:

```bash
anypinn create my-project --template sir --data synthetic --lightning
```

??? example "Full CLI output"

    ```
    ● anypinn v0.x.x
    │
    ◇ Generating project in 'my-project'...
    │
    ◇ Choose a starting point
    │  SIR Epidemic Model
    │
    ◇ Select training data source
    │  Generate synthetic data
    │
    ◇ Include Lightning training wrapper?
    │  Yes
    │
    ◇ Creating 'my-project/'
    │  ├ pyproject.toml    project dependencies
    │  ├ ode.py            mathematical definition
    │  ├ config.py         training configuration
    │  ├ train.py          execution script
    │  └ data/             data directory
    │
    ◇ Dependencies synced
    │
    ● Starting training with `uv run train.py`...
       You can stop at any time and it will save the current checkpoint!
    ```

---

## Project structure

The generated project contains five items:

```
my-project/
├── pyproject.toml    # Dependencies (anypinn, torch, numpy, scipy, pandas, ...)
├── ode.py            # ODE definition, fields, parameters, validation
├── config.py         # All hyperparameters in a single frozen dataclass
├── train.py          # Training script (Lightning or plain PyTorch)
└── data/             # Empty (for CSV data or saved predictions)
```

### `ode.py`: the physics

This is the file you will modify most. It contains:

- **The ODE callable**: a function `f(x, y, args) -> Tensor` that returns
  derivatives. `x` is the independent variable (e.g. time), `y` is the state
  vector, and `args` is an `ArgsRegistry` holding both fixed constants and
  learnable parameters.
- **`create_problem(hp)`**: a factory that wires up `Field`s (neural network
  solution approximations), `Parameter`s (quantities to recover), and the ODE
  into an `ODEInverseProblem`.
- **`validation`**: a `ValidationRegistry` mapping parameter names to
  ground-truth callables. The library logs MSE against these at every training
  step.

For the SIR template, the ODE looks like this:

```python
def SIR(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    S, I = y
    b, d, N = args["beta"], args["delta"], args["N"]

    dS = -b(x) * I * S * C / N(x)
    dI = b(x) * I * S * C / N(x) - d(x) * I

    return torch.stack([dS * T, dI * T])
```

`beta` is a `Parameter` (learnable); `delta` and `N` are `Argument`s (fixed).
The PINN recovers `beta` by minimizing the residual of the ODE against observed
data.

### `config.py`: hyperparameters

All training hyperparameters live in a single `ODEHyperparameters` dataclass:

```python
hp = ODEHyperparameters(
    lr=5e-4,
    max_epochs=2000,
    gradient_clip_val=0.1,
    training_data=GenerationConfig(...),
    fields_config=MLPConfig(hidden_layers=[64, 128, 128, 64], ...),
    params_config=MLPConfig(hidden_layers=[64, 128, 128, 64], ...),
    scheduler=ReduceLROnPlateauConfig(factor=0.5, patience=55, ...),
    pde_weight=1,
    ic_weight=1,
    data_weight=1,
)
```

The config is frozen and keyword-only. Changing a hyperparameter means changing
one value here, and the rest of the project picks it up automatically.

### `train.py`: execution

With the Lightning wrapper, `train.py` sets up loggers, callbacks, and a
`Trainer`. With the core loop, it is a standard PyTorch training script. Both
support `--predict` to load a saved checkpoint and run inference without
retraining.

---

## Train the model

```bash
cd my-project
uv sync
uv run train.py
```

`uv sync` resolves and installs all dependencies from `pyproject.toml`.
`uv run train.py` starts training. With the Lightning wrapper, you will see a
progress bar with live loss values.

Monitor training in TensorBoard:

```bash
uv run tensorboard --logdir logs/tensorboard
```

---

## What happens next

Once training completes, see [Understanding the Output](understanding-the-output.md)
to learn what the generated files mean, then [Next Steps](next-steps.md) for
where to go from here.
