# TODO — Architecture & Scalability Audit

Analysis of `anypinn` covering scaling, performance, developer experience, and ODE/PDE expansion readiness. Organised by priority within each category.

---

## 1. Scaling

### ~~S1. Hard-coded `num_workers=7` in DataLoaders~~ ✅

**File:** `core/dataset.py:238,250`

~~Both `train_dataloader()` and `predict_dataloader()` fix `num_workers=7`. This is machine-specific — it will over-subscribe on a 4-core laptop and under-utilise a 64-core server. Workers should be configurable (or default to `os.cpu_count()`).~~

**Resolved:** Defaulted `num_workers` to `cpu_count() or 1`.

### ~~S2. Repeated device transfers in validation lookups~~ ✅

**File:** `core/validation.py:114-116`

~~`make_lookup_fn` calls `values.to(x.device)` on every forward pass. The tensor should be moved once (e.g. via a `register_buffer` pattern or a device-aware closure) instead of per-call.~~

**Resolved:** Added per-device cache in `make_lookup_fn` closure.

### ~~S3. Scalar `Argument` allocates a new tensor every call~~ ✅

**File:** `core/nn.py:146`

~~`torch.tensor(self._value, device=x.device)` in `Argument.__call__` creates a fresh tensor each forward pass. For fixed-value arguments called millions of times per training run, cache the tensor per device.~~

**Resolved:** Added per-device `_tensor_cache` in `Argument`.

### ~~S4. Collocation generator instantiates a new `torch.Generator` per batch~~ ✅

**File:** `core/dataset.py:113`

~~`torch.Generator().manual_seed(idx)` is allocated in `__getitem__`. A single reusable generator re-seeded each call would avoid repeated object allocation at scale.~~

**Resolved:** Created `self._coll_gen` once in `__init__` and re-seed it each call.

### ~~S5. No GPU memory pinning or prefetching~~ ✅

**File:** `core/dataset.py:235-252`

~~Neither `pin_memory=True` nor any prefetch factor is set on the DataLoaders. For GPU training this leaves easy bandwidth on the table.~~

**Resolved:** Added `pin_memory=True` to both `train_dataloader` and `predict_dataloader`.

### ~~S6. `ICConstraint` moves tensors to device every loss call~~ ✅

**File:** `problems/ode.py:141-142`

~~`self.t0.to(device)` and `self.Y0.to(device)` inside `loss()` re-run every step. These should be moved once at `inject_context` time or registered as buffers.~~

**Resolved:** Tensors are moved to device on first `loss()` call and cached for subsequent calls.

### ~~S7. Loss accumulator creates an unnecessary zero tensor~~ ✅

**File:** `core/problem.py:107`

~~`torch.tensor(0.0, device=...)` starts the accumulator. Starting from the first constraint's loss and summing the rest avoids one extra tensor + addition.~~

**Resolved:** Start accumulation from the first constraint's loss; only allocate a zero tensor for the empty-constraints edge case.

---

## 2. Performance

### ~~P1. Per-field autograd loop in `ResidualsConstraint`~~ ✅

**File:** `problems/ode.py:90-95`

~~Gradients are computed one field at a time in a list comprehension:~~

```python
dy_dt = torch.stack([
    torch.autograd.grad(pred, x_coll, torch.ones_like(pred), create_graph=True)[0]
    for pred in preds
])
```

~~This creates N separate backward graphs. Computing a single batched Jacobian (via `torch.vmap` + `jacrev`, or stacking outputs first) would be faster and use less memory for systems with many state variables.~~

**Resolved:** Batched autograd calls in `ResidualsConstraint.loss`.

### ~~P2. `DataScaling` recomputes min/max without caching~~ ✅

**File:** `lightning/callbacks.py:192-194`

~~`x.min()` and `x.max()` are called multiple times within `transform_data`. The results should be computed once and stored.~~

**Resolved:** Cached min/max in `DataScaling.transform_data`.

### ~~P3. Only Adam optimizer with ReduceLROnPlateau~~ ✅

**File:** `lightning/module.py:88-110`

~~The optimizer and scheduler are hard-coded. L-BFGS is a well-known strong performer for PINN training (often converges in 10x fewer iterations). Users should be able to select the optimiser and scheduler family via config.~~

**Resolved:** Configurable optimizer and scheduler.

### ~~P4. `SMMAStopping` uses a Python list as a ring buffer~~ ✅

**File:** `lightning/callbacks.py:72-73`

~~`self.smma_buffer.pop(0)` is O(n). For large lookback windows, use `collections.deque(maxlen=...)`.~~

**Resolved:** Replaced list with `collections.deque(maxlen=lookback)`, removing manual `pop(0)`.

### ~~P5. Validation loss computed during training even if only logged~~ ✅

**File:** `core/problem.py:112-115`

~~`_param_validation_loss` runs a full forward + MSE against ground truth every step inside `training_loss`. This is diagnostic, not part of the optimised loss, but it still appears on the backward graph. Consider detaching or computing it only every N epochs.~~

**Resolved:** Wrapped `_param_validation_loss` with `@torch.no_grad()` to avoid unnecessary graph construction.

---

## 3. Developer Experience (DX)

### ~~D1. Assertions instead of `ValueError` / `TypeError`~~ ✅

**File:** `core/dataset.py:198-204`, `core/nn.py:35`, `core/dataset.py:62,65,68`

~~All shape and value checks use `assert`, which is stripped by `python -O`. These should be proper `ValueError` raises with descriptive messages explaining the expected vs. actual shape.~~

**Resolved:** Replaced all `assert` statements with `ValueError`, `TypeError`, and `RuntimeError` with descriptive messages in `nn.py` and `dataset.py`.

### ~~D2. No config validation in `__post_init__`~~ ✅

**File:** `core/config.py` (all dataclasses)

~~`lr`, `batch_size`, `data_ratio`, `collocations`, etc. are never validated at construction time. A negative `lr` or zero `batch_size` only surfaces as a cryptic runtime error deep in training. Add `__post_init__` guards.~~

**Resolved:** Added `__post_init__` validation to `AdamConfig`, `LBFGSConfig`, `ReduceLROnPlateauConfig`, `CosineAnnealingConfig`, `EarlyStoppingConfig`, `SMMAStoppingConfig`, `TrainingDataConfig`, and `PINNHyperparameters`.

### ~~D3. Registry key mismatches are silent~~ ✅

**Files:** `core/problem.py`, `problems/ode.py`

~~If the keys in `FieldsRegistry` don't match the outputs expected by the ODE callable, the error surfaces as a shape mismatch or `KeyError` deep in autograd. A factory or builder that validates registry keys against the ODE signature would catch this at construction time.~~

**Resolved:** Added `ODEProperties.expected_args` (optional `frozenset[str]`) and construction-time validation of field count vs y0 length in `ResidualsConstraint` and `ICConstraint`. When `expected_args` is set, missing keys are reported immediately.

### ~~D4. Squeeze/unsqueeze assumptions are fragile~~ ✅

**File:** `core/problem.py:135-138`

~~`f(x).squeeze(-1)` silently drops dimensions. If a Field accidentally outputs shape `(N, 1, 1)`, `squeeze(-1)` produces `(N, 1)` — different from the `(N,)` expected downstream. Explicit reshape with documented shape contracts would be safer.~~

**Resolved:** Added `reshape(n, -1)` before every `squeeze(-1)` on field/param outputs.

### ~~D5. `ColumnRef` resolution assumes evenly-spaced integer indices~~ ✅

**File:** `core/validation.py:115`

```python
idx = x.squeeze(-1).round().to(torch.int32)
```

~~This only works when x values are integer indices (0, 1, 2, ...). For continuous or irregularly-spaced x it silently returns wrong values. Needs interpolation or an explicit index-mapping strategy.~~

**Resolved:** `make_lookup_fn` now returns a `_ColumnLookup` wrapper. `Problem._get_true_param` detects it and converts domain coordinates to row indices via `(x - x0) / dx` using the inferred domain before calling the lookup. Pure callable validation sources receive actual coordinates unchanged.

### ~~D6. `ColumnRef` re-reads CSV on every resolution~~ ✅

**File:** `core/validation.py:99`

~~Each `ColumnRef` in the registry triggers a fresh `pd.read_csv(df_path)`. If there are 5 validated parameters, the CSV is read 5 times. Read once and share.~~

**Resolved:** DataFrame cached in a local variable before the loop; `pd.read_csv` called at most once.

### ~~D7. `y` shape handling in `load_data` is inconsistent~~ ✅

**File:** `core/dataset.py:157-158`

```python
if y.shape[1] != 1:
y = y.unsqueeze(-1)

```

~~For a single-column y with shape `(N, 1)`, the condition is false and no unsqueeze happens — correct. But for `y` with shape `(N,)` (1-D), the `y.shape[1]` access raises an `IndexError` before the guard can act.~~

**Resolved:** Replaced with `if y.ndim == 1` — handles 1-D tensors safely and doesn't unsqueeze multi-column `(N, k)` output.

### ~~D8. No "hello world" minimal example~~ ✅

**Files:** `examples/`

~~Every example uses the full Lightning stack with scaling, SMMA stopping, custom progress bars. A 20-line example using only `anypinn.core` with a manual training loop would dramatically lower the onboarding barrier.~~

**Resolved:** Added `examples/exponential_decay/exponential_decay.py` — ~80 lines, pure PyTorch loop, no Lightning, no catalog. Learns decay rate k in dy/dt = -ky with analytic ground truth.

---

## 4. PDE Expansion Readiness

### ~~PDE1. `Domain1D` is the only domain representation — Critical~~ ✅

~~**File:** `core/nn.py:18-44`~~

~~The domain is hard-coded as a 1-D interval `[x0, x1]` with scalar step `dx`. PDEs require multi-dimensional domains (rectangles, circles, irregular meshes). A `Domain` base class with `Domain1D`, `Domain2D`, `DomainND` subclasses is needed. This cascades through:~~

~~- `InferredContext` (holds a single `Domain1D`)~~
~~- `PINNDataModule.gen_coll()` signature (takes `Domain1D`)~~
~~- `PINNDataset` shape assertions (hard-code `shape[1] == 1`)~~

### ~~PDE2. No boundary condition abstraction~~

~~**File:** `problems/ode.py`~~

~~The three constraint types (`ResidualsConstraint`, `ICConstraint`, `DataConstraint`) are ODE-specific. PDEs need:~~

~~- **Dirichlet** BC: `u(x) = g(x)` on boundary~~
~~- **Neumann** BC: `du/dn = h(x)` on boundary~~
~~- **Robin / periodic / mixed** BCs~~

~~These should live in a new `anypinn.problems.pde` module as `Constraint` subclasses, each receiving a boundary region and a target function.~~ ✅

### ~~PDE3. No higher-order or mixed derivative utilities~~ ✅

~~**File:** `problems/ode.py:90-95`~~

~~Only first-order temporal derivatives (`dy/dt`) are computed. PDEs commonly need:~~

~~- Second-order: `d²u/dx²` (heat, wave equations)~~
~~- Mixed partials: `d²u/dxdy`~~
~~- Laplacian: `nabla²u`~~

~~A differential operator utility (e.g. `grad(u, x, order=2)`, `laplacian(u, coords)`) would make PDE constraints composable without re-implementing autograd boilerplate in every constraint.~~

**Resolved:** Added `lib/diff.py` with composable operators (`grad`, `partial`, `mixed_partial`, `laplacian`, `divergence`, `hessian`). Refactored `ResidualsConstraint` and `NeumannBCConstraint` to use the shared utilities.

### PDE4. Collocation generation is 1-D only

**Files:** `core/dataset.py:167`, `catalog/*.py`

`gen_coll(domain: Domain1D) -> Tensor` produces shape `(M, 1)`. Multi-dimensional PDEs need `(M, d)` collocation points over complex geometries. The collocation strategy should be decoupled into a `CollocationSampler` protocol with implementations:

- `UniformSampler` (grid)
- `RandomSampler` (uniform random)
- `LatinHypercubeSampler`
- `AdaptiveSampler` (residual-based refinement)

### ~~PDE5. Shape assertions block multi-dimensional inputs~~ ✅

~~**File:** `core/dataset.py:199-204`~~

```python
assert x_data.shape[1] == 1, "x shape differs than (n, 1)."
assert self.coll.shape[1] == 1, "coll shape differs than (m, 1)."
```

~~These must be relaxed to `shape[1] == d` where `d` is the spatial dimension. Otherwise no PDE can pass through `PINNDataModule.setup()`.~~

**Resolved:** Assertions replaced with `ValueError` checks allowing `d >= 1` and enforcing `x_data.shape[1] == coll.shape[1]`. Test coverage added in `tests/test_dataset.py::TestPINNDataModuleSetup`.

### PDE6. `Field` input encoding assumes 1-D

**File:** `core/nn.py:82-83`

`MLPConfig.in_dim` defaults to 1 in all examples and catalog problems. Fourier feature encoding (`encode` callback) is supported but there are no built-in spatial encodings (positional encoding, random Fourier features) useful for high-frequency PDE solutions.

### PDE7. `ODEInverseProblem` hardcodes `MSELoss`

**File:** `problems/ode.py:247`

For multi-scale PDEs (e.g. reaction-diffusion with stiff terms), MSE can be dominated by the largest residual component. The criterion should be configurable (Huber, weighted MSE, relative L2, etc.).

### PDE8. No multi-output / coupled-system support pattern

**File:** `core/problem.py`

`Problem` applies all constraints to all fields uniformly. Coupled PDEs (e.g. Navier-Stokes: velocity + pressure) need constraints that operate on subsets of fields — e.g. continuity constraint on pressure only, momentum on velocity only. Currently there's no way to scope a constraint to specific fields without manually filtering inside each constraint.

### ODE1. No native second-order ODE constraint (requires first-order state augmentation)

**Files:** `problems/ode.py`, `examples/damped_oscillator/damped_oscillator.py`

Second-order ODEs are currently modeled by introducing auxiliary state variables and rewriting to a first-order system (e.g. `x'' -> [x', v']` with `v = x'`). This works, but it increases model complexity and learns derivative relationships indirectly.

Add a native second-order ODE path:

- `SecondOrderResidualsConstraint` using `lib/diff.py::partial(..., order=2)`
- Second-order callable protocol (receives `x`, `y`, `dy/dx`, args and returns `d2y/dx2`)
- Initial-derivative condition support (`y(t0)` and `dy/dx(t0)`)
- Optional `SecondOrderODEInverseProblem` wrapper for parity with `ODEInverseProblem`

---

## Summary Matrix

| ID     | Category  | Severity   | Effort     | Impact                                      |
| ------ | --------- | ---------- | ---------- | ------------------------------------------- |
| ~~P1~~ | ~~Perf~~  | ~~High~~   | ~~Medium~~ | ~~2-3x residual training speedup~~ ✅       |
| ~~PDE1~~ | ~~PDE~~ | ~~Critical~~ | ~~Large~~ | ~~Blocks all PDE work~~ ✅                |
| ~~PDE2~~ | ~~PDE~~ | ~~Critical~~ | ~~Large~~ | ~~Blocks all PDE work~~ ✅                |
| ~~PDE5~~ | ~~PDE~~ | ~~Critical~~ | ~~Small~~ | ~~Blocks all PDE work~~ ✅                |
| ~~D1~~ | ~~DX~~    | ~~High~~   | ~~Small~~  | ~~Silent production failures~~ ✅           |
| ~~D2~~ | ~~DX~~    | ~~High~~   | ~~Small~~  | ~~Bad error messages~~ ✅                   |
| ~~S1~~ | ~~Scale~~ | ~~Medium~~ | ~~Small~~  | ~~Portability across hardware~~ ✅          |
| ~~S2~~ | ~~Scale~~ | ~~Medium~~ | ~~Small~~  | ~~O(n) wasted device transfers~~ ✅         |
| ~~D5~~ | ~~DX~~    | ~~Medium~~ | ~~Medium~~ | ~~Wrong validation on non-integer data~~ ✅ |
| ~~PDE3~~ | ~~PDE~~ | ~~High~~ | ~~Medium~~ | ~~Core utility for PDE constraints~~ ✅    |
| PDE4   | PDE       | High       | Medium     | Needed for any 2D+ problem                  |
| ~~P3~~ | ~~Perf~~  | ~~Medium~~ | ~~Medium~~ | ~~L-BFGS convergence gains~~ ✅             |
| ~~D3~~ | ~~DX~~    | ~~Medium~~ | ~~Medium~~ | ~~Prevents misconfiguration~~ ✅            |
| ~~S3~~ | ~~Scale~~ | ~~Low~~    | ~~Small~~  | ~~Minor allocation overhead~~ ✅            |
| ~~S4~~ | ~~Scale~~ | ~~Low~~    | ~~Small~~  | ~~Minor allocation overhead~~ ✅            |
| ~~S5~~ | ~~Scale~~ | ~~Low~~    | ~~Small~~  | ~~Easy GPU bandwidth win~~ ✅               |
| ~~S6~~ | ~~Scale~~ | ~~Low~~    | ~~Small~~  | ~~Minor per-step overhead~~ ✅              |
| ~~S7~~ | ~~Scale~~ | ~~Low~~    | ~~Small~~  | ~~Trivial~~ ✅                              |
| ~~P2~~ | ~~Perf~~  | ~~Low~~    | ~~Small~~  | ~~Minor~~ ✅                                |
| ~~P4~~ | ~~Perf~~  | ~~Low~~    | ~~Small~~  | ~~Only matters for huge lookback~~ ✅       |
| ~~P5~~ | ~~Perf~~  | ~~Low~~    | ~~Small~~  | ~~Diagnostic overhead~~ ✅                  |
| ~~D6~~ | ~~DX~~    | ~~Low~~    | ~~Small~~  | ~~Repeated I/O~~ ✅                         |
| ~~D7~~ | ~~DX~~    | ~~Low~~    | ~~Small~~  | ~~Edge-case crash~~ ✅                      |
| ~~D8~~ | ~~DX~~    | ~~Medium~~ | ~~Small~~  | ~~Onboarding~~ ✅                           |
| PDE6   | PDE       | Medium     | Medium     | Quality of PDE solutions                    |
| PDE7   | PDE       | Medium     | Small      | Multi-scale PDE accuracy                    |
| PDE8   | PDE       | Medium     | Large      | Coupled-system expressiveness               |
| ODE1   | ODE       | Medium     | Medium     | Native 2nd-order ODE expressiveness         |
