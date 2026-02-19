"""
Minimal anypinn example — learn the decay rate k in dy/dt = -k·y, y(0) = 1.

No Lightning. No catalog. Uses anypinn.core + anypinn.problems only.
Run: uv run python examples/exponential_decay/exponential_decay.py
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from anypinn.core import (
    ArgsRegistry,
    Field,
    GenerationConfig,
    InferredContext,
    MLPConfig,
    Parameter,
    PINNDataset,
    ScalarConfig,
)
from anypinn.problems import ODEHyperparameters, ODEInverseProblem, ODEProperties

# ── 1. Ground-truth data (analytic solution) ─────────────────────────────────
K_TRUE = 0.7
t = torch.linspace(0, 5, 60).unsqueeze(-1)  # (60, 1)
y = torch.exp(-K_TRUE * t)                   # (60, 1)
t_coll = torch.rand(500, 1) * 5             # (500, 1) collocation points

# ── 2. Dataset + DataLoader ───────────────────────────────────────────────────
dataset = PINNDataset(x_data=t, y_data=y, x_coll=t_coll, batch_size=30, data_ratio=10)
loader = DataLoader(dataset, batch_size=None, num_workers=0)

# ── 3. ODE: dy/dt = -k·y ─────────────────────────────────────────────────────
def exp_decay(x: Tensor, y: Tensor, args: ArgsRegistry) -> Tensor:
    (y_hat,) = y
    return torch.stack([-args["k"](x) * y_hat])

# ── 4. Fields, parameters, problem ───────────────────────────────────────────
mlp = MLPConfig(in_dim=1, out_dim=1, hidden_layers=[32, 32], activation="tanh")
fields: dict = {"y": Field(mlp)}
params: dict = {"k": Parameter(ScalarConfig(init_value=0.5))}

props = ODEProperties(ode=exp_decay, y0=torch.tensor([1.0]), args={})

hp = ODEHyperparameters(
    lr=1e-3,
    training_data=GenerationConfig(
        batch_size=30,
        data_ratio=10,
        collocations=500,
        x=t.squeeze(),
        noise_level=0.0,
        args_to_train={},
    ),
    fields_config=mlp,
    params_config=ScalarConfig(init_value=0.5),
    pde_weight=1.0,
    ic_weight=1.0,
    data_weight=1.0,
)

problem = ODEInverseProblem(
    props=props,
    hp=hp,
    fields=fields,
    params=params,
    predict_data=lambda x, f, _: f["y"](x),
)

# ── 5. Inject context (domain bounds + optional validation) ───────────────────
problem.inject_context(InferredContext(x=t, y=y, validation={}))

# ── 6. Training loop ──────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(problem.parameters(), lr=hp.lr)

for epoch in range(200):
    for batch in loader:
        optimizer.zero_grad()
        loss = problem.training_loss(batch)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 50 == 0:
        k_hat = params["k"](t[:1]).item()
        print(
            f"epoch {epoch + 1:4d}  loss={loss.item():.3e}"
            f"  k_learned={k_hat:.4f}  k_true={K_TRUE}"
        )
