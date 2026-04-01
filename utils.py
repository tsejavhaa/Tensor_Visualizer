"""
Utility helpers for visualize-tensor.
"""

import numpy as np


def _to_numpy(tensor):
    if hasattr(tensor, "detach"):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def tensor_summary(tensor) -> str:
    """
    Return a human-readable summary string for any tensor.

    Parameters
    ----------
    tensor : torch.Tensor or ndarray

    Returns
    -------
    str

    Examples
    --------
    >>> import torch
    >>> from visualize_tensor import tensor_summary
    >>> t = torch.rand(10, 10, 3)
    >>> print(tensor_summary(t))
    """
    arr  = _to_numpy(tensor).astype(float)
    flat = arr.ravel()

    dtype = arr.dtype
    lines = [
        "─" * 42,
        f"  visualize-tensor  |  shape: {arr.shape}",
        "─" * 42,
        f"  dtype   : {dtype}",
        f"  ndim    : {arr.ndim}",
        f"  numel   : {flat.size:,}",
        f"  min     : {flat.min():.6f}",
        f"  max     : {flat.max():.6f}",
        f"  mean    : {flat.mean():.6f}",
        f"  std     : {flat.std():.6f}",
        f"  sum     : {flat.sum():.6f}",
        "─" * 42,
    ]
    result = "\n".join(lines)
    print(result)
    return result