"""Differential operators for Physics-Informed Neural Networks.

Composable utilities built on ``torch.autograd.grad`` for computing
first-order, higher-order, and mixed partial derivatives without
re-implementing autograd boilerplate in every constraint.

All operators default to ``create_graph=True`` so their outputs are
differentiable — required when used inside loss functions that must
back-propagate. Pass ``create_graph=False`` for detached results
(e.g. visualisation or adaptive sampling).

.. note::

    For even higher performance on large-batch Hessians or Jacobians,
    ``torch.func.jacrev`` / ``torch.func.hessian`` can be composed with
    ``torch.vmap``. The operators here intentionally use the simpler
    ``autograd.grad`` path for broad compatibility and ``torch.compile``
    friendliness.
"""

import torch
from torch import Tensor


def grad(
    u: Tensor,
    x: Tensor,
    *,
    create_graph: bool = True,
) -> Tensor:
    """Compute the full gradient ∇u with respect to coordinates *x*.

    Args:
        u: Scalar field values, shape ``(n,)`` or ``(n, 1)``.
        x: Input coordinates, shape ``(n, d)`` with ``requires_grad=True``.
        create_graph: Keep the result in the computation graph (default ``True``).

    Returns:
        Tensor of shape ``(n, d)`` — ``[∂u/∂x_0, …, ∂u/∂x_{d-1}]`` per sample.
    """
    (grad_u,) = torch.autograd.grad(
        u.reshape(-1).sum(),
        x,
        create_graph=create_graph,
    )
    return grad_u


def partial(
    u: Tensor,
    x: Tensor,
    dim: int,
    *,
    order: int = 1,
    create_graph: bool = True,
) -> Tensor:
    """Compute ∂\\ :sup:`order` u / ∂x\\ :sub:`dim`\\ :sup:`order`.

    Args:
        u: Scalar field values, shape ``(n,)`` or ``(n, 1)``.
        x: Input coordinates, shape ``(n, d)`` with ``requires_grad=True``.
        dim: Spatial dimension index to differentiate along.
        order: Derivative order (≥ 1, default 1).
        create_graph: Keep the result in the computation graph (default ``True``).

    Returns:
        Tensor of shape ``(n, 1)``.
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    result = grad(u, x, create_graph=True)[:, dim : dim + 1]
    for _ in range(order - 1):
        result = grad(result, x, create_graph=True)[:, dim : dim + 1]

    if not create_graph:
        result = result.detach()
    return result


def mixed_partial(
    u: Tensor,
    x: Tensor,
    dims: tuple[int, ...],
    *,
    create_graph: bool = True,
) -> Tensor:
    """Compute ∂\\ :sup:`k` u / (∂x\\ :sub:`d₀` ∂x\\ :sub:`d₁` ⋯).

    Derivatives are applied left-to-right: first differentiate w.r.t.
    ``dims[0]``, then the result w.r.t. ``dims[1]``, and so on.

    Args:
        u: Scalar field values, shape ``(n,)`` or ``(n, 1)``.
        x: Input coordinates, shape ``(n, d)`` with ``requires_grad=True``.
        dims: Dimension indices to differentiate along, in order.
        create_graph: Keep the result in the computation graph (default ``True``).

    Returns:
        Tensor of shape ``(n, 1)``.
    """
    if len(dims) == 0:
        raise ValueError("dims must contain at least one dimension index")

    result = grad(u, x, create_graph=True)[:, dims[0] : dims[0] + 1]
    for d in dims[1:]:
        result = grad(result, x, create_graph=True)[:, d : d + 1]

    if not create_graph:
        result = result.detach()
    return result


def laplacian(
    u: Tensor,
    x: Tensor,
    *,
    create_graph: bool = True,
) -> Tensor:
    """Compute the Laplacian ∇²u = Σᵢ ∂²u/∂xᵢ².

    Computes the full first-order gradient once and then differentiates
    each component — ``d + 1`` autograd calls total for ``d`` dimensions.

    Args:
        u: Scalar field values, shape ``(n,)`` or ``(n, 1)``.
        x: Input coordinates, shape ``(n, d)`` with ``requires_grad=True``.
        create_graph: Keep the result in the computation graph (default ``True``).

    Returns:
        Tensor of shape ``(n, 1)``.
    """
    grad_u = grad(u, x, create_graph=True)
    ndim = x.shape[1]
    lap = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
    for i in range(ndim):
        d2u_dxi2 = grad(grad_u[:, i : i + 1], x, create_graph=True)[:, i : i + 1]
        lap = lap + d2u_dxi2

    if not create_graph:
        lap = lap.detach()
    return lap


def divergence(
    v: Tensor,
    x: Tensor,
    *,
    create_graph: bool = True,
) -> Tensor:
    """Compute the divergence ∇·v = Σᵢ ∂vᵢ/∂xᵢ.

    Args:
        v: Vector field values, shape ``(n, d)`` matching ``x.shape[1]``.
        x: Input coordinates, shape ``(n, d)`` with ``requires_grad=True``.
        create_graph: Keep the result in the computation graph (default ``True``).

    Returns:
        Tensor of shape ``(n, 1)``.
    """
    ndim = x.shape[1]
    if v.shape[1] != ndim:
        raise ValueError(
            f"v has {v.shape[1]} components but x has {ndim} dimensions; they must match"
        )

    div = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
    for i in range(ndim):
        dvi_dxi = grad(v[:, i : i + 1], x, create_graph=True)[:, i : i + 1]
        div = div + dvi_dxi

    if not create_graph:
        div = div.detach()
    return div


def hessian(
    u: Tensor,
    x: Tensor,
    *,
    create_graph: bool = True,
) -> Tensor:
    """Compute the Hessian matrix H[u] where H\\ :sub:`ij` = ∂²u/∂xᵢ∂xⱼ.

    Computes the first-order gradient once, then differentiates each
    component to build each row — ``d + 1`` autograd calls total.

    Args:
        u: Scalar field values, shape ``(n,)`` or ``(n, 1)``.
        x: Input coordinates, shape ``(n, d)`` with ``requires_grad=True``.
        create_graph: Keep the result in the computation graph (default ``True``).

    Returns:
        Tensor of shape ``(n, d, d)``.
    """
    grad_u = grad(u, x, create_graph=True)
    ndim = x.shape[1]
    rows = []
    for i in range(ndim):
        row = grad(grad_u[:, i : i + 1], x, create_graph=True)
        rows.append(row)

    H = torch.stack(rows, dim=1)

    if not create_graph:
        H = H.detach()
    return H
