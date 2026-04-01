"""
Core visualization engine for visualize-tensor.
Inspired by the stacked-layer visualization style for 3D tensors.
"""

import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patheffects as pe


def _safe_tight_layout(fig=None, **kwargs):
    """Apply tight layout while silencing incompatible-axes warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if fig is None:
            fig = plt.gcf()
        fig.tight_layout(**kwargs)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _to_numpy(tensor):
    """Convert a PyTorch tensor (or numpy array) to a numpy array."""
    if hasattr(tensor, "detach"):          # torch.Tensor
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def _fmt(val, decimals=2):
    """Format a scalar value for display inside a cell."""
    return f"{val:.{decimals}f}"


def _layer_colors():
    """Return (face, edge, text) colour triples for each depth layer."""
    return [
        ("#FFFFFF", "#333333", "#222222"),   # front  – white
        ("#EEF2FF", "#5B6AD0", "#3B4CC0"),   # mid    – lavender
        ("#E0FAF0", "#2DA870", "#1A7A50"),   # back   – mint
        ("#FFF4E0", "#E08A00", "#A05800"),   # extra  – amber
    ]


# ─── show() ────────────────────────────────────────────────────────────────────

def show(tensor, title=None, decimals=2, figsize=None, cmap="coolwarm",
         show_values=True, cell_size=0.7):
    """
    Visualize a tensor with an auto-selected style.

    Parameters
    ----------
    tensor      : torch.Tensor or numpy.ndarray
    title       : Optional title string
    decimals    : Decimal places shown in cells
    figsize     : (width, height) override
    cmap        : Matplotlib colormap name (used for value-to-colour mapping)
    show_values : Whether to print numeric values inside cells
    cell_size   : Visual size of each cell (inches)

    Examples
    --------
    >>> import torch
    >>> from visualize_tensor import show
    >>> t = torch.rand(10, 10, 3)
    >>> show(t)
    """
    arr = _to_numpy(tensor)

    if arr.ndim == 1:
        _show_1d(arr, title, decimals, figsize, cmap, show_values, cell_size)
    elif arr.ndim == 2:
        _show_2d(arr, title, decimals, figsize, cmap, show_values, cell_size)
    elif arr.ndim == 3:
        show_layers(tensor, title=title, decimals=decimals, figsize=figsize,
                    show_values=show_values, cell_size=cell_size)
    else:
        show_stats(tensor, title=title)


# ─── show_layers() ─────────────────────────────────────────────────────────────

def _truncated_indices(size, head=4, tail=4, threshold=10):
    """
    Return (indices, ellipsis_pos) for a dimension of `size`.

    If size <= threshold  → show all indices, ellipsis_pos = None
    Else                  → show first `head` + last `tail` indices,
                            ellipsis_pos = head  (slot index where "..." sits)

    The returned indices list already includes the real row/col indices in
    display order.  The caller inserts a visual "…" gap at ellipsis_pos.
    """
    if size <= threshold:
        return list(range(size)), None
    return list(range(head)) + list(range(size - tail, size)), head


def show_layers(tensor, title=None, decimals=2, figsize=None,
                show_values=True, cell_size=0.65, max_layers=4,
                head=4, tail=4, threshold=10):
    """
    Visualize a 3-D tensor as stacked, offset grid layers.

    For tensors with more than `threshold` rows or columns the display is
    automatically truncated to the first `head` + last `tail` rows/cols with
    a clear "·····" gap in the middle — so large tensors stay readable.

    Parameters
    ----------
    tensor    : torch.Tensor or ndarray  — shape (H, W, D)
    title     : Optional figure title
    decimals  : Decimal places per cell
    figsize   : (w, h) override
    show_values : Show numeric values inside cells
    cell_size : Base cell size in inches
    max_layers : Cap the number of visible depth layers
    head      : Number of rows/cols shown at the start before "·····"
    tail      : Number of rows/cols shown at the end  after  "·····"
    threshold : Tensors with rows or cols > threshold trigger truncation

    Examples
    --------
    >>> import torch
    >>> from visualize_tensor import show_layers
    >>> t = torch.rand(10, 10, 3)
    >>> show_layers(t)
    >>> t_big = torch.rand(32, 32, 3)
    >>> show_layers(t_big)          # shows 4 + ····· + 4 rows & cols
    >>> show_layers(t_big, head=5, tail=5)   # customise head/tail count
    """
    arr = _to_numpy(tensor)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    if arr.ndim != 3:
        raise ValueError(f"show_layers expects a 3-D tensor, got shape {arr.shape}")

    true_rows, true_cols, true_depth = arr.shape
    depth = min(true_depth, max_layers)

    # ── Build truncated index lists ──────────────────────────────────────────
    row_indices, row_ellipsis = _truncated_indices(true_rows, head, tail, threshold)
    col_indices, col_ellipsis = _truncated_indices(true_cols, head, tail, threshold)

    # Display dimensions: real cells + optional ellipsis slot (half a cell wide)
    ELIPSIS_FRAC = 0.55        # ellipsis column/row is this fraction of cell_size
    n_rows_vis = len(row_indices) + (1 if row_ellipsis is not None else 0)
    n_cols_vis = len(col_indices) + (1 if col_ellipsis is not None else 0)

    grid_w = (len(col_indices) * cell_size +
              (ELIPSIS_FRAC * cell_size if col_ellipsis is not None else 0))
    grid_h = (len(row_indices) * cell_size +
              (ELIPSIS_FRAC * cell_size if row_ellipsis is not None else 0))

    # ── Layout constants ────────────────────────────────────────────────────
    OFFSET_X = 0.9
    OFFSET_Y = 0.55
    PADDING  = 0.35
    FONT     = max(5.5, min(9.0, cell_size * 11))
    EFONT    = max(6.0, cell_size * 10)       # font for "·····" label

    total_w = grid_w + OFFSET_X * (depth - 1) + 2 * PADDING + 1.2
    total_h = grid_h + OFFSET_Y * (depth - 1) + 2 * PADDING + (0.6 if title else 0.2)

    if figsize is None:
        figsize = (max(total_w, 5), max(total_h, 4))

    fig = plt.figure(figsize=figsize, facecolor="#FAFAFA")
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, figsize[0])
    ax.set_ylim(0, figsize[1])
    ax.axis("off")
    ax.set_facecolor("#FAFAFA")

    colors = _layer_colors()
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmin == vmax:
        vmin, vmax = vmin - 0.5, vmax + 0.5
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap("Blues")

    # ── Helper: map display slot → x/y offset in inches ─────────────────────
    def col_x(slot):
        """Left edge of display column slot (0-based), accounting for ellipsis."""
        if col_ellipsis is None or slot < col_ellipsis:
            return slot * cell_size
        elif slot == col_ellipsis:
            return col_ellipsis * cell_size          # ellipsis slot start
        else:
            return col_ellipsis * cell_size + ELIPSIS_FRAC * cell_size + (slot - col_ellipsis - 1) * cell_size

    def row_y(slot):
        """Bottom edge of display row slot (0-based from bottom), accounting for ellipsis."""
        if row_ellipsis is None or slot < row_ellipsis:
            return slot * cell_size
        elif slot == row_ellipsis:
            return row_ellipsis * cell_size
        else:
            return row_ellipsis * cell_size + ELIPSIS_FRAC * cell_size + (slot - row_ellipsis - 1) * cell_size

    # ── Draw layers back → front ─────────────────────────────────────────────
    for d in range(depth - 1, -1, -1):
        fc, ec, tc = colors[d % len(colors)]
        ox = PADDING + d * OFFSET_X
        oy = PADDING + (depth - 1 - d) * OFFSET_Y

        # Bounding box
        box = FancyBboxPatch(
            (ox - 0.04, oy - 0.04),
            grid_w + 0.08, grid_h + 0.08,
            boxstyle="round,pad=0.06",
            linewidth=1.5 if d == 0 else 1.0,
            edgecolor=ec, facecolor=fc, alpha=0.82,
            zorder=d * 10,
        )
        ax.add_patch(box)

        # Depth label
        ax.text(
            ox + grid_w + 0.12, oy + grid_h / 2,
            f"[:, :, {d}]",
            fontsize=6.5, color=ec, va="center", ha="left",
            fontfamily="monospace", alpha=0.75, zorder=d * 10 + 5,
        )

        # ── Draw cells ───────────────────────────────────────────────────────
        # Build display slot lists: real slots interleaved with ellipsis slot
        row_slots = list(range(len(row_indices)))
        if row_ellipsis is not None:
            row_slots = list(range(row_ellipsis)) + [None] + list(range(row_ellipsis, len(row_indices)))

        col_slots = list(range(len(col_indices)))
        if col_ellipsis is not None:
            col_slots = list(range(col_ellipsis)) + [None] + list(range(col_ellipsis, len(col_indices)))

        n_display_rows = len(row_slots)
        n_display_cols = len(col_slots)

        for ri, rslot in enumerate(row_slots):
            for ci, cslot in enumerate(col_slots):

                # ── Ellipsis cell (row AND col gap) ──────────────────────────
                if rslot is None and cslot is None:
                    cx = ox + col_x(ci) + ELIPSIS_FRAC * cell_size / 2
                    cy = oy + row_y(n_display_rows - 1 - ri) + ELIPSIS_FRAC * cell_size / 2
                    ax.text(cx, cy, "⋱", ha="center", va="center",
                            fontsize=EFONT, color=ec, alpha=0.4,
                            fontfamily="monospace", zorder=d * 10 + 2)
                    continue

                # ── Ellipsis row gap ──────────────────────────────────────────
                if rslot is None:
                    cw = ELIPSIS_FRAC * cell_size if cslot == col_ellipsis else cell_size
                    cx = ox + col_x(ci) + cw / 2
                    cy = oy + row_y(n_display_rows - 1 - ri) + ELIPSIS_FRAC * cell_size / 2
                    if d == 0:
                        ax.text(cx, cy, "·····", ha="center", va="center",
                                fontsize=EFONT, color=ec, alpha=0.45,
                                fontfamily="monospace", zorder=d * 10 + 2)
                    continue

                # ── Ellipsis col gap ──────────────────────────────────────────
                if cslot is None:
                    ch = ELIPSIS_FRAC * cell_size if rslot == row_ellipsis else cell_size
                    cx = ox + col_x(ci) + ELIPSIS_FRAC * cell_size / 2
                    cy = oy + row_y(n_display_rows - 1 - ri) + cell_size / 2
                    if d == 0:
                        ax.text(cx, cy, "⋮", ha="center", va="center",
                                fontsize=EFONT, color=ec, alpha=0.45,
                                fontfamily="monospace", zorder=d * 10 + 2)
                    continue

                # ── Normal cell ───────────────────────────────────────────────
                real_r = row_indices[rslot]
                real_c = col_indices[cslot]
                val    = arr[real_r, real_c, d]

                cell_x = ox + col_x(ci)
                cell_y = oy + row_y(n_display_rows - 1 - ri)
                cx     = cell_x + cell_size / 2
                cy     = cell_y + cell_size / 2

                if d == 0:
                    tint = cmap_obj(norm(val) * 0.35 + 0.05)
                    ax.add_patch(FancyBboxPatch(
                        (cell_x + 0.03, cell_y + 0.03),
                        cell_size - 0.06, cell_size - 0.06,
                        boxstyle="round,pad=0.02",
                        linewidth=0, facecolor=tint, alpha=0.45,
                        zorder=d * 10 + 1,
                    ))

                if show_values:
                    ax.text(
                        cx, cy, _fmt(val, decimals),
                        ha="center", va="center",
                        fontsize=FONT, color=tc,
                        fontweight="bold" if d == 0 else "normal",
                        fontfamily="monospace",
                        zorder=d * 10 + 2,
                    )

        # ── Grid lines (skip across ellipsis gaps) ───────────────────────────
        lw, lalp = 0.5, 0.3
        # horizontal lines
        for ri in range(n_display_rows + 1):
            if ri < n_display_rows and row_slots[ri] is None:
                continue   # don't draw a full line through the ellipsis row
            y_  = oy + row_y(ri) if ri < n_display_rows else oy + grid_h
            ax.plot([ox, ox + grid_w], [y_, y_],
                    color=ec, lw=lw, alpha=lalp, zorder=d * 10 + 1)
        # vertical lines
        for ci in range(n_display_cols + 1):
            if ci < n_display_cols and col_slots[ci] is None:
                continue
            x_ = ox + col_x(ci) if ci < n_display_cols else ox + grid_w
            ax.plot([x_, x_], [oy, oy + grid_h],
                    color=ec, lw=lw, alpha=lalp, zorder=d * 10 + 1)

    # ── Title ────────────────────────────────────────────────────────────────
    real_shape = (true_rows, true_cols, true_depth)
    label = title or f"Tensor  {real_shape}"
    if row_ellipsis is not None or col_ellipsis is not None:
        label += f"  [showing {head}+…+{tail}]"
    ax.text(
        figsize[0] / 2, figsize[1] - 0.22, label,
        ha="center", va="top",
        fontsize=11, fontweight="bold",
        color="#222222", fontfamily="monospace",
        zorder=999,
    )

    _safe_tight_layout(pad=0)
    plt.show()


# ─── show_heatmap() ────────────────────────────────────────────────────────────

def show_heatmap(tensor, title=None, decimals=2, figsize=None, cmap="viridis",
                 show_values=True):
    """
    Visualize a 2-D tensor (or each channel of a 3-D tensor) as a colour heatmap.

    Parameters
    ----------
    tensor    : torch.Tensor or ndarray
    title     : Optional title
    decimals  : Decimal places shown in cells
    figsize   : (w, h) override
    cmap      : Matplotlib colormap name
    show_values : Print values inside cells

    Examples
    --------
    >>> import torch
    >>> from visualize_tensor import show_heatmap
    >>> t = torch.rand(8, 8)
    >>> show_heatmap(t, title="My 8×8 tensor", cmap="plasma")
    """
    arr = _to_numpy(tensor)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    if arr.ndim != 3:
        raise ValueError(f"show_heatmap expects ≤3-D tensor, got {arr.shape}")

    rows, cols, depth = arr.shape
    fig, axes = plt.subplots(1, depth, figsize=figsize or (max(4, cols * 0.6) * depth, max(4, rows * 0.6)))
    if depth == 1:
        axes = [axes]

    fig.patch.set_facecolor("#FAFAFA")
    vmin, vmax = arr.min(), arr.max()

    for d, ax in enumerate(axes):
        layer = arr[:, :, d]
        im = ax.imshow(layer, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_title(f"[:, :, {d}]" if depth > 1 else (title or f"{arr.shape[:2]}"),
                     fontsize=9, fontfamily="monospace")
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.tick_params(labelsize=7)

        if show_values:
            font_size = max(5, min(9, 60 // max(rows, cols)))
            for r in range(rows):
                for c in range(cols):
                    val   = layer[r, c]
                    color = "white" if (val - vmin) / (vmax - vmin + 1e-9) > 0.55 else "black"
                    ax.text(c, r, _fmt(val, decimals), ha="center", va="center",
                            fontsize=font_size, color=color, fontfamily="monospace")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title or f"Tensor {arr.shape}", fontsize=11, fontweight="bold",
                 fontfamily="monospace", y=1.02)
    _safe_tight_layout()
    plt.show()


# ─── show_stats() ──────────────────────────────────────────────────────────────

def show_stats(tensor, title=None, figsize=(9, 5)):
    """
    Show a statistical summary panel for any-shaped tensor:
    distribution histogram, per-channel stats table, and a shape diagram.

    Parameters
    ----------
    tensor  : torch.Tensor or ndarray
    title   : Optional title
    figsize : (w, h) override

    Examples
    --------
    >>> import torch
    >>> from visualize_tensor import show_stats
    >>> t = torch.randn(64, 128)
    >>> show_stats(t)
    """
    arr  = _to_numpy(tensor).astype(float)
    flat = arr.ravel()

    fig = plt.figure(figsize=figsize, facecolor="#FAFAFA")
    fig.suptitle(title or f"Tensor Stats  {arr.shape}", fontsize=12,
                 fontweight="bold", fontfamily="monospace")

    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.38)

    # ── Histogram ───────────────────────────────────────────────────────────
    ax_hist = fig.add_subplot(gs[:, 0])
    ax_hist.hist(flat, bins=min(40, len(flat) // 4 or 10), color="#5B6AD0",
                 edgecolor="white", linewidth=0.4, alpha=0.85, orientation="horizontal")
    ax_hist.set_xlabel("Count", fontsize=8)
    ax_hist.set_ylabel("Value", fontsize=8)
    ax_hist.set_title("Distribution", fontsize=9, fontfamily="monospace")
    ax_hist.tick_params(labelsize=7)
    ax_hist.set_facecolor("#F0F2FF")

    # ── Stats table ─────────────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[:, 1:])
    ax_tbl.axis("off")

    if arr.ndim >= 3:
        depth  = arr.shape[2]
        labels = [f"[:, :, {d}]" for d in range(min(depth, 8))]
        slices = [arr[:, :, d].ravel() for d in range(min(depth, 8))]
    elif arr.ndim == 2:
        labels = ["full"]
        slices = [flat]
    else:
        labels = ["full"]
        slices = [flat]

    rows_data = []
    for lbl, sl in zip(labels, slices):
        rows_data.append([
            lbl,
            f"{sl.mean():.4f}",
            f"{sl.std():.4f}",
            f"{sl.min():.4f}",
            f"{sl.max():.4f}",
        ])

    col_labels = ["Slice", "Mean", "Std", "Min", "Max"]
    tbl = ax_tbl.table(
        cellText=rows_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.55)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#5B6AD0")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EEF2FF")
        else:
            cell.set_facecolor("#FAFAFA")
        cell.set_edgecolor("#DDDDDD")

    ax_tbl.set_title("Per-slice Statistics", fontsize=9, fontfamily="monospace",
                     pad=4)

    _safe_tight_layout()
    plt.show()


# ─── 1-D and 2-D internal helpers ──────────────────────────────────────────────

def _show_1d(arr, title, decimals, figsize, cmap, show_values, cell_size):
    """Render a 1-D tensor as a single-row grid."""
    n = len(arr)
    fw = figsize[0] if figsize else max(6, n * cell_size + 1)
    fig, ax = plt.subplots(figsize=(fw, max(2, cell_size + 1)), facecolor="#FAFAFA")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")
    ax.set_facecolor("#FAFAFA")
    vmin, vmax = arr.min(), arr.max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)
    for i, v in enumerate(arr):
        color = cmap_obj(norm(v))
        rect  = plt.Rectangle((i - 0.5, -0.5), 1, 1, color=color, alpha=0.5,
                               linewidth=1, edgecolor="#555")
        ax.add_patch(rect)
        if show_values:
            ax.text(i, 0, _fmt(v, decimals), ha="center", va="center",
                    fontsize=8, fontfamily="monospace", fontweight="bold")
    ax.set_title(title or f"Tensor  {arr.shape}", fontsize=10, fontfamily="monospace")
    _safe_tight_layout()
    plt.show()


def _show_2d(arr, title, decimals, figsize, cmap, show_values, cell_size):
    """Render a 2-D tensor via show_heatmap."""
    show_heatmap(arr, title=title, decimals=decimals, figsize=figsize,
                 cmap=cmap, show_values=show_values)