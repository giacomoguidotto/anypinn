# Use CSV Data

This guide shows how to replace synthetic data generation with real
experimental observations loaded from a CSV file.

---

## Scaffold with CSV mode

```bash
anypinn create my-project --template sir --data csv --lightning
```

The `--data csv` flag generates an `IngestionConfig` instead of a
`GenerationConfig` in `config.py`.

---

## Prepare your data

Place your CSV file in the `data/` directory. The file must have:

- **A column for the independent variable** (e.g. time)
- **One or more columns for observed quantities** (e.g. infected counts)

Example `data/observations.csv`:

```csv
t,I_obs
0.0,1.0
1.0,3.2
2.0,8.7
3.0,21.4
...
```

!!! warning "Column order matters"

    The first column must be the independent variable. Remaining columns are
    treated as observed field values in the order listed in `y_columns`.

---

## Configure data loading

In `config.py`, update the `IngestionConfig` to point to your file:

```python
from pathlib import Path
from anypinn.core import IngestionConfig

hp = ODEHyperparameters(
    ...
    training_data=IngestionConfig(
        batch_size=100,
        data_ratio=2,              # Ratio of data points to collocation points per batch
        collocations=6000,
        df_path=Path("./data/observations.csv"),
        y_columns=["I_obs"],       # Which columns contain observed values
    ),
    ...
)
```

| Field | Meaning |
| ----- | ------- |
| `df_path` | Path to the CSV file (relative to the project root) |
| `y_columns` | List of column names to use as observation data |
| `collocations` | Number of collocation points for physics enforcement |
| `data_ratio` | How many data points per collocation point in each batch |
| `batch_size` | Points per training batch |

---

## Match fields to columns

In `ode.py`, the `predict_data` function maps neural field outputs to the
observed quantities. Make sure the output order matches `y_columns`:

```python
def predict_data(x_data: Tensor, fields: FieldsRegistry, _params: ParamsRegistry) -> Tensor:
    I_pred = fields["I"](x_data)
    return I_pred.unsqueeze(1)  # Shape: (n_points, 1, 1) — one column
```

If you observe multiple quantities:

```python
def predict_data(x_data, fields, _params):
    S_pred = fields["S"](x_data)
    I_pred = fields["I"](x_data)
    return torch.stack([S_pred, I_pred], dim=1)  # (n_points, 2, 1)
```

---

## Data callbacks

Use `DataCallback` subclasses to transform data before training. Common use
cases:

- **Normalization** — scale observations to a manageable range
- **Time rescaling** — map the time domain to `[0, 1]`

The built-in `DataScaling` callback handles common scaling:

```python
from anypinn.lightning.callbacks import DataScaling

data_module = MyDataModule(
    hp=hp,
    callbacks=[DataScaling(y_scale=1 / 1e6)],  # Scale down large populations
)
```

---

## Validation with real data

When using real data, you may not have ground-truth parameter values for
validation. In that case, pass an empty validation registry:

```python
validation: ValidationRegistry = {}
```

The training will still log total loss and constraint-level losses, but
parameter MSE metrics will not appear.

If you do have reference values (e.g. from literature), provide them:

```python
validation: ValidationRegistry = {
    "beta": 0.3,  # Literature value for comparison
}
```
