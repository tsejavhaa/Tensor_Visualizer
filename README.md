# visualize-tensor

Friendly tensor visualization for PyTorch beginners. Instead of staring at raw numbers, you can render tensors as layered grids, heatmaps, and quick statistical summaries.

`visualize-tensor` works with both PyTorch tensors and NumPy arrays.

## What it does

- Auto-selects a readable view based on tensor shape
- Renders 1D tensors as a labeled row
- Renders 2D tensors as heatmaps
- Renders 3D tensors as stacked layer views or per-channel heatmaps
- Prints fast console summaries for shape, dtype, and value ranges
- Includes a CLI for demoing or visualizing saved `.pt`, `.pth`, `.npy`, and `.npz` files
- Includes Jupyter helpers for richer inline tensor display

## Install

Base install:

```bash
pip install visualize-tensor
```

With PyTorch support:

```bash
pip install "visualize-tensor[torch]"
```

From a local wheel:

```bash
pip install visualize_tensor-0.3.0-py3-none-any.whl
```

## Quick start

```python
import torch
from visualize_tensor import show, show_layers, show_heatmap, show_stats, tensor_summary

t1 = torch.rand(12)
t2 = torch.rand(8, 8)
t3 = torch.rand(10, 10, 3)

show(t1, title="1D tensor")
show(t2, title="2D tensor")
show(t3, title="3D tensor")

show_layers(t3, title="Stacked layers")
show_heatmap(t2, title="Heatmap", cmap="viridis")
show_stats(t3, title="Tensor stats")

tensor_summary(t3)
```

## Main API

### `show(tensor, ...)`

Auto-selects a visualization style:

- `1D` -> labeled row view
- `2D` -> heatmap
- `3D` -> stacked layer view
- `4D+` -> stats summary

Useful when you just want a sensible default.

### `show_layers(tensor, ...)`

Best for `H x W x D` tensors. Layers are drawn as offset panels so each channel is visible. Large tensors are truncated automatically with a visible gap in the middle to keep the plot readable.

Example:

```python
import torch
from visualize_tensor import show_layers

t = torch.rand(32, 32, 3)
show_layers(t, title="Feature map", head=4, tail=4)
```

### `show_heatmap(tensor, ...)`

Displays a 2D tensor as a heatmap. If given a 3D tensor, it renders one heatmap per channel.

```python
import torch
from visualize_tensor import show_heatmap

t = torch.rand(8, 8)
show_heatmap(t, title="Activation map", cmap="plasma")
```

### `show_stats(tensor, ...)`

Shows a histogram and summary statistics for any tensor shape.

```python
import torch
from visualize_tensor import show_stats

t = torch.randn(64, 128)
show_stats(t, title="Distribution overview")
```

### `tensor_summary(tensor)`

Prints and returns a compact text summary:

- shape
- dtype
- ndim
- number of elements
- min / max
- mean / std / sum

```python
from visualize_tensor import tensor_summary
tensor_summary([[1, 2], [3, 4]])
```

## CLI

The package ships with two console commands:

```bash
visualize-tensor --help
vt --help
```

Run a quick demo:

```bash
visualize-tensor demo
visualize-tensor demo --mode heatmap
visualize-tensor demo --mode stats
```

Visualize a saved file:

```bash
visualize-tensor file my_tensor.pt
visualize-tensor file my_array.npy
visualize-tensor file my_archive.npz --mode heatmap
```

Print package info:

```bash
visualize-tensor info
```

Supported file types:

- `.pt`
- `.pth`
- `.npy`
- `.npz`

## Jupyter notebook support

Import the notebook helper once:

```python
import visualize_tensor.notebook
```

After that, evaluating a PyTorch tensor in a notebook cell will use a richer visual representation when possible.

## Why this project exists

Tensor tooling is often optimized for experts. This project aims to make tensors easier to inspect while learning deep learning, debugging model outputs, or teaching tensor shapes to others.

## Development

This repository includes:

- `core.py` for visualization logic
- `utils.py` for summary helpers
- `demo.py` for a quick tour
- `test_clean.ipynb` for notebook-based examples

Install development dependencies:

```bash
pip install ".[dev]"
```

Run the demo locally:

```bash
python demo.py
```

Build a wheel:

```bash
python -m build
```

## Requirements

- Python `>=3.8`
- `numpy>=1.21`
- `matplotlib>=3.5`
- Optional: `torch>=1.10`

## License

MIT
