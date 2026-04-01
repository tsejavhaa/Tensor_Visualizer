"""
examples/demo.py  —  visualize-tensor quick tour
Run:  python examples/demo.py
"""

import numpy as np
import torch

# Use numpy arrays if torch isn't installed
try:
    t3d  = torch.rand(14, 14, 3)
    t2d  = torch.rand(8, 8)
    t1d  = torch.rand(12)
    tbig = torch.randn(32, 64)
    print("Using PyTorch tensors")
except ImportError:
    rng  = np.random.default_rng(42)
    t3d  = rng.random((10, 10, 3))
    t2d  = rng.random((8, 8))
    t1d  = rng.random(12)
    tbig = rng.standard_normal((32, 64))
    print("PyTorch not found — using NumPy arrays")

from visualize_tensor import show, show_layers, show_heatmap, show_stats, tensor_summary, show_layers

# ── 1. Auto show (3-D) ───────────────────────────────────────────────────────
print("\n▶  show()  →  3-D tensor")
show(t3d, title="Random 10×10×3 Tensor")

# ── 2. Explicit layers ───────────────────────────────────────────────────────
print("\n▶  show_layers()")
show_layers(t3d, title="Stacked Layer View")

# ── 3. Heatmap 2-D ───────────────────────────────────────────────────────────
print("\n▶  show_heatmap()  →  2-D")
show_heatmap(t2d, title="8×8 Heatmap", cmap="viridis")

# ── 4. Heatmap 3-D (per channel) ─────────────────────────────────────────────
print("\n▶  show_heatmap()  →  3-D (per channel)")
show_heatmap(t3d, title="Per-channel heatmaps", cmap="plasma")

# ── 5. Stats panel ───────────────────────────────────────────────────────────
print("\n▶  show_stats()")
show_stats(tbig, title="32×64 stats")

# ── 6. Console summary ───────────────────────────────────────────────────────
print("\n▶  tensor_summary()")
tensor_summary(t3d)