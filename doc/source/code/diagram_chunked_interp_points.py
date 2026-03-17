"""Script to generate a diagram for chunked raster-to-point interpolation."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle


# -----------------------------------------------------------------------------
# Example data
# -----------------------------------------------------------------------------

# Raster grid
NROWS = 8
NCOLS = 10
CHUNK_ROWS = (4, 4)
CHUNK_COLS = (5, 5)

# Interpolation method and implied overlap depth
METHOD = "cubic"
DEPTH = 4

SELECTED_CHUNK_LW = 2.8
EXAMPLE_CHUNK_LW = 4.2

# Input point coordinates in raster data coordinates
POINT_X = np.array([0.9, 1.8, 3.7, 4.4, 1.4, 6.2, 8.4, 8.7, 2.6, 6.9])
POINT_Y = np.array([6.7, 5.2, 6.0, 2.2, 1.4, 5.6, 2.3, 1.3, 1.8, 2.7])

# Conceptual chunking of the input point ribbon
POINT_CHUNKS = (4, 3, 3)
HIGHLIGHT_POINT_CHUNK = 1

point_chunk_starts = np.concatenate([[0], np.cumsum(POINT_CHUNKS)])
POINT_IDS_HIGHLIGHT = np.arange(
    point_chunk_starts[HIGHLIGHT_POINT_CHUNK],
    point_chunk_starts[HIGHLIGHT_POINT_CHUNK + 1],
)


def point_to_raster_chunk(x: float, y: float) -> tuple[int, int]:
    """Assign point to raster chunk based on containing chunk."""
    col_edges = np.concatenate([[0], np.cumsum(CHUNK_COLS)])
    row_edges_top = np.concatenate([[0], np.cumsum(CHUNK_ROWS)])

    ix = np.searchsorted(col_edges[1:], x, side="right")
    row_from_top = np.searchsorted(row_edges_top[1:], NROWS - y, side="right")
    iy = row_from_top
    return iy, ix


POINT_CHUNK_LOC = [point_to_raster_chunk(float(x), float(y)) for x, y in zip(POINT_X, POINT_Y, strict=True)]
HIGHLIGHTED_RASTER_CHUNKS = sorted({POINT_CHUNK_LOC[i] for i in POINT_IDS_HIGHLIGHT})
highlighted_raster_chunk = POINT_CHUNK_LOC[POINT_IDS_HIGHLIGHT[0]]

# Example interpolated values for the output ribbon
OUTPUT_VALUES = np.array([12.4, 15.0, 14.1, 9.8, 11.3, 10.7, 13.2, 8.9, 7.4, 10.1])


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------

NEUTRAL = "#333333"
PIXEL_GRID = "0.80"
CHUNK_COLOR = "#222222"
HIGHLIGHT_COLOR = "#F58518"
POINT_COLOR = "#6BAED6"
POINT_EDGE = "#4C78A8"
POINT_FADED = "#BFD7EA"
OVERLAP_FILL_ALPHA = 0.18


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def raster_chunk_bounds(chunk_loc: tuple[int, int]) -> tuple[float, float, float, float]:
    """Return chunk bounds as (left, bottom, right, top)."""
    iy, ix = chunk_loc

    row_starts = np.concatenate([[0], np.cumsum(CHUNK_ROWS)])
    col_starts = np.concatenate([[0], np.cumsum(CHUNK_COLS)])

    top_row = row_starts[iy]
    bottom_row = row_starts[iy + 1]
    left_col = col_starts[ix]
    right_col = col_starts[ix + 1]

    left = float(left_col)
    right = float(right_col)
    top = float(NROWS - top_row)
    bottom = float(NROWS - bottom_row)

    return left, bottom, right, top


def expanded_chunk_bounds(
    bounds: tuple[float, float, float, float],
    *,
    depth_px: float,
) -> tuple[float, float, float, float]:
    """Expand chunk bounds by interpolation overlap depth."""
    left, bottom, right, top = bounds
    return left - depth_px, bottom - depth_px, right + depth_px, top + depth_px


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def _setup_axis(ax: plt.Axes, *, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    """Common axis formatting."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_pixel_grid(ax: plt.Axes, *, nrows: int, ncols: int, extend: float = 1.0) -> None:
    """Draw thin raster pixel grid, slightly extended around the raster."""
    segments: list[np.ndarray] = []

    for x in range(ncols + 1):
        segments.append(np.array([[x, -extend], [x, nrows + extend]], dtype=float))

    for y in range(nrows + 1):
        segments.append(np.array([[-extend, y], [ncols + extend, y]], dtype=float))

    coll = LineCollection(
        segments,
        colors=PIXEL_GRID,
        linewidths=0.8,
        capstyle="round",
        joinstyle="round",
        zorder=1,
        clip_on=False,
    )
    ax.add_collection(coll)


def _draw_chunk_boundaries(ax: plt.Axes, *, extend: float = 1.0) -> None:
    """Draw all raster chunk boundaries."""
    row_starts = np.concatenate([[0], np.cumsum(CHUNK_ROWS)])
    col_starts = np.concatenate([[0], np.cumsum(CHUNK_COLS)])

    for x in col_starts:
        ax.plot(
            [x, x],
            [-extend, NROWS + extend],
            color=CHUNK_COLOR,
            linewidth=2.3,
            zorder=3,
            solid_capstyle="round",
            clip_on=False,
        )

    for y_idx in row_starts:
        y = NROWS - y_idx
        ax.plot(
            [-extend, NCOLS + extend],
            [y, y],
            color=CHUNK_COLOR,
            linewidth=2.3,
            zorder=3,
            solid_capstyle="round",
            clip_on=False,
        )


def _draw_points_on_raster(
    ax: plt.Axes,
    *,
    highlight_ids: np.ndarray | None = None,
    show_ids: bool = False,
) -> None:
    """Draw points in raster coordinates."""
    highlight_set = set([] if highlight_ids is None else highlight_ids.tolist())

    for i, (x, y) in enumerate(zip(POINT_X, POINT_Y, strict=True)):
        highlighted = i in highlight_set
        ax.scatter(
            x,
            y,
            s=42 if highlighted else 34,
            facecolor=POINT_COLOR if highlighted else POINT_FADED,
            edgecolor=POINT_EDGE,
            linewidth=1.0,
            alpha=0.95 if highlighted else 0.75,
            zorder=6 if highlighted else 4,
        )
        if show_ids:
            ax.text(
                x + 0.12,
                y + 0.12,
                f"{i}",
                fontsize=8.8,
                color=NEUTRAL,
                ha="left",
                va="bottom",
                zorder=7,
            )


def _draw_highlight_chunk(
    ax: plt.Axes,
    bounds: tuple[float, float, float, float],
    *,
    color: str,
    lw: float = 2.8,
) -> None:
    """Draw highlighted chunk outline."""
    left, bottom, right, top = bounds
    rect = Rectangle(
        (left, bottom),
        right - left,
        top - bottom,
        fill=False,
        edgecolor=color,
        linewidth=lw,
        zorder=7,
    )
    ax.add_patch(rect)


def _draw_overlap_chunk(ax: plt.Axes, bounds: tuple[float, float, float, float]) -> None:
    """Draw overlap-expanded source chunk."""
    left, bottom, right, top = bounds
    fill = Rectangle(
        (left, bottom),
        right - left,
        top - bottom,
        facecolor=HIGHLIGHT_COLOR,
        edgecolor="none",
        alpha=OVERLAP_FILL_ALPHA,
        zorder=2,
        clip_on=False,
    )
    edge = Rectangle(
        (left, bottom),
        right - left,
        top - bottom,
        fill=False,
        edgecolor=HIGHLIGHT_COLOR,
        linewidth=2.6,
        zorder=7,
        clip_on=False,
    )
    ax.add_patch(fill)
    ax.add_patch(edge)


def _draw_point_ribbon(
    ax: plt.Axes,
    *,
    highlight_chunk: int,
    values: np.ndarray | None = None,
    title: str | None = None,
) -> None:
    """Draw a 1D point ribbon with chunk separators."""
    total = len(POINT_X)
    y_center = 0.0
    box_h = 0.9

    starts = np.concatenate([[0], np.cumsum(POINT_CHUNKS)])

    rect = Rectangle(
        (0, y_center - 0.5 * box_h),
        total,
        box_h,
        fill=False,
        edgecolor=CHUNK_COLOR,
        linewidth=2.2,
        zorder=3,
    )
    ax.add_patch(rect)

    for s in starts[1:-1]:
        ax.plot([s, s], [y_center - 0.5 * box_h, y_center + 0.5 * box_h], color=CHUNK_COLOR, lw=2.2, zorder=3)

    x0 = starts[highlight_chunk]
    x1 = starts[highlight_chunk + 1]

    fill = Rectangle(
        (x0, y_center - 0.5 * box_h),
        x1 - x0,
        box_h,
        facecolor=HIGHLIGHT_COLOR,
        edgecolor="none",
        alpha=0.12,
        zorder=1,
    )
    edge = Rectangle(
        (x0, y_center - 0.5 * box_h),
        x1 - x0,
        box_h,
        fill=False,
        edgecolor=HIGHLIGHT_COLOR,
        linewidth=2.6,
        zorder=4,
    )
    ax.add_patch(fill)
    ax.add_patch(edge)

    for i in range(total):
        x = i + 0.5
        highlighted = x0 <= i < x1
        ax.scatter(
            x,
            y_center,
            s=42 if highlighted else 34,
            facecolor=POINT_COLOR if highlighted else POINT_FADED,
            edgecolor=POINT_EDGE,
            linewidth=1.0,
            alpha=0.95 if highlighted else 0.75,
            zorder=5,
        )

        label = f"{i}" if values is None else f"{values[i]:.1f}"
        ax.text(
            x,
            y_center - 0.62,
            label,
            ha="center",
            va="top",
            fontsize=8.5,
            color=NEUTRAL,
            zorder=6,
        )

    if title is not None:
        ax.text(
            x0 + 0.5 * (x1 - x0),
            y_center + 0.72,
            title,
            ha="center",
            va="bottom",
            fontsize=10,
            color=HIGHLIGHT_COLOR,
            fontweight="bold",
        )


def _add_workflow_arrow(
    fig: plt.Figure,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    rad: float = 0.0,
) -> None:
    """Add a figure-level workflow arrow."""
    arrow = FancyArrowPatch(
        start,
        end,
        transform=fig.transFigure,
        arrowstyle="-|>",
        connectionstyle=f"arc3,rad={rad}",
        linewidth=1.8,
        color=NEUTRAL,
        mutation_scale=18,
        zorder=30,
    )
    fig.add_artist(arrow)

def _add_visual_legend(fig: plt.Figure) -> None:
    """Draw a two-row horizontal legend with balanced spacing."""
    y1 = 0.05
    y2 = 0.018

    centers = [0.1, 0.35, 0.6, 0.8]

    sym_w = 0.032
    txt_gap = 0.012

    # ---------------- Row 1 ----------------
    # Pixel grid
    cx = centers[0]
    x = cx - 0.075
    fig.add_artist(
        Line2D(
            [x, x + sym_w],
            [y1, y1],
            transform=fig.transFigure,
            color=PIXEL_GRID,
            lw=0.8,
            solid_capstyle="round",
        )
    )
    fig.text(x + sym_w + txt_gap, y1, "Raster pixel grid", transform=fig.transFigure, va="center", fontsize=10)

    # Chunk boundary
    cx = centers[1]
    x = cx - 0.085
    fig.add_artist(
        Line2D(
            [x, x + sym_w],
            [y1, y1],
            transform=fig.transFigure,
            color=CHUNK_COLOR,
            lw=2.3,
            solid_capstyle="round",
        )
    )
    fig.text(x + sym_w + txt_gap, y1, "Raster chunk boundary", transform=fig.transFigure, va="center", fontsize=10)

    # All input points (light blue)
    cx = centers[2]
    x = cx - 0.080
    fig.add_artist(
        Line2D(
            [x + sym_w / 2],
            [y1],
            transform=fig.transFigure,
            marker="o",
            markersize=7,
            markerfacecolor=POINT_FADED,
            markeredgecolor=POINT_EDGE,
            linestyle="None",
        )
    )
    fig.text(x + sym_w + txt_gap, y1, "All input points", transform=fig.transFigure, va="center", fontsize=10)

    # Selected point chunk (darker blue)
    cx = centers[3]
    x = cx - 0.090
    fig.add_artist(
        Line2D(
            [x + sym_w / 2],
            [y1],
            transform=fig.transFigure,
            marker="o",
            markersize=7,
            markerfacecolor=POINT_COLOR,
            markeredgecolor=POINT_EDGE,
            linestyle="None",
        )
    )
    fig.text(x + sym_w + txt_gap, y1, "Selected point chunk", transform=fig.transFigure, va="center", fontsize=10)

    # ---------------- Row 2 ----------------
    # Selected raster chunks
    cx = centers[1]
    x = cx - 0.25
    rect1 = Rectangle(
        (x, y2 - 0.010),
        sym_w,
        0.020,
        transform=fig.transFigure,
        fill=False,
        edgecolor=HIGHLIGHT_COLOR,
        linewidth=SELECTED_CHUNK_LW,
    )
    fig.add_artist(rect1)
    fig.text(x + sym_w + txt_gap, y2, "Raster chunks used by this point chunk", transform=fig.transFigure,
             va="center", fontsize=10)

    # Example raster chunk / overlap
    cx = centers[3]
    x = cx - 0.25
    rect_fill = Rectangle(
        (x, y2 - 0.010),
        sym_w,
        0.020,
        transform=fig.transFigure,
        facecolor=HIGHLIGHT_COLOR,
        edgecolor="none",
        alpha=0.15,
    )
    rect_edge = Rectangle(
        (x, y2 - 0.010),
        sym_w,
        0.020,
        transform=fig.transFigure,
        fill=False,
        edgecolor=HIGHLIGHT_COLOR,
        linewidth=EXAMPLE_CHUNK_LW,
    )
    fig.add_artist(rect_fill)
    fig.add_artist(rect_edge)
    fig.text(x + sym_w + txt_gap, y2, "Example chunk processed with overlap", transform=fig.transFigure, va="center", fontsize=10)


def _add_left_column_separator(fig: plt.Figure, ax_top_left: plt.Axes, ax_bottom_left: plt.Axes) -> None:
    """Add a subtle horizontal separator in the left column."""
    pos_t = ax_top_left.get_position()
    pos_b = ax_bottom_left.get_position()

    x0 = pos_t.x0 - 0.2 * (pos_t.x1 - pos_t.x0)
    x1 = pos_t.x1 + 0.2 * (pos_t.x1 - pos_t.x0)
    y = 0.52 * (pos_t.y0 + pos_b.y1)

    line = Line2D(
        [x0, x1],
        [y, y],
        transform=fig.transFigure,
        color="0.75",
        lw=2.5,
        alpha=0.8,
        solid_capstyle="round",
        zorder=50,
    )
    fig.add_artist(line)

def _draw_highlight_chunks(
    ax: plt.Axes,
    bounds_list: list[tuple[float, float, float, float]],
    *,
    color: str,
    example_bounds: tuple[float, float, float, float] | None = None,
    lw_selected: float = SELECTED_CHUNK_LW,
    lw_example: float = EXAMPLE_CHUNK_LW,
) -> None:
    """Draw selected raster chunks, with one optional example chunk thicker."""
    for bounds in bounds_list:
        left, bottom, right, top = bounds

        is_example = example_bounds is not None and np.allclose(bounds, example_bounds)
        lw = lw_example if is_example else lw_selected

        rect = Rectangle(
            (left, bottom),
            right - left,
            top - bottom,
            fill=False,
            edgecolor=color,
            linewidth=lw,
            zorder=7,
            joinstyle="round",
        )
        ax.add_patch(rect)

# -----------------------------------------------------------------------------
# Main figure
# -----------------------------------------------------------------------------

def make_chunked_interp_points_diagram() -> tuple[plt.Figure, np.ndarray]:
    """Build a 4-panel schematic for chunked raster-to-point interpolation."""
    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.30)

    ax_ul = fig.add_subplot(gs[0, 0])  # Input points
    ax_ur = fig.add_subplot(gs[0, 1])  # Assign to raster chunks
    ax_bl = fig.add_subplot(gs[1, 0])  # Reordered outputs
    ax_br = fig.add_subplot(gs[1, 1])  # Interpolate on overlap-expanded chunks

    axes = np.array([ax_ul, ax_ur, ax_bl, ax_br], dtype=object)

    chunk_bounds_list = [raster_chunk_bounds(ch) for ch in HIGHLIGHTED_RASTER_CHUNKS]
    chunk_bounds = max(chunk_bounds_list, key=lambda b: (b[3], b[
        2]))  # Use upper-right selected chunk
    overlap_bounds = expanded_chunk_bounds(chunk_bounds, depth_px=1.0)

    # -------------------------------------------------------------------------
    # Upper left: input points
    # -------------------------------------------------------------------------
    _draw_point_ribbon(ax_ul, highlight_chunk=HIGHLIGHT_POINT_CHUNK, title="Point chunk")
    ax_ul.text(
        0.5,
        1.02,
        "Input points",
        transform=ax_ul.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=NEUTRAL,
    )
    ax_ul.text(
        0.5,
        0,
        "Points are chunked along the 1D input sequence",
        transform=ax_ul.transAxes,
        ha="center",
        va="top",
        fontsize=10.2,
        color=NEUTRAL,
    )
    _setup_axis(ax_ul, xlim=(-0.4, len(POINT_X) + 0.4), ylim=(-1.0, 1.4))
    ax_ul.set_aspect("auto")

    # -------------------------------------------------------------------------
    # Upper right: assign to raster chunks
    # -------------------------------------------------------------------------
    _draw_pixel_grid(ax_ur, nrows=NROWS, ncols=NCOLS, extend=1.0)
    _draw_chunk_boundaries(ax_ur, extend=1.0)
    _draw_points_on_raster(ax_ur, highlight_ids=POINT_IDS_HIGHLIGHT, show_ids=True)
    _draw_highlight_chunks(
        ax_ur,
        chunk_bounds_list,
        color=HIGHLIGHT_COLOR,
        example_bounds=chunk_bounds,
    )
    ax_ur.text(
        0.5,
        1.02,
        "Assign points to raster chunks",
        transform=ax_ur.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=NEUTRAL,
    )
    ax_ur.text(
        0.5,
        -0.05,
        "Map point chunk coordinates to\nall intersecting raster chunks",
        transform=ax_ur.transAxes,
        ha="center",
        va="top",
        fontsize=10.2,
        color=NEUTRAL,
        linespacing=1.25,
    )
    _setup_axis(ax_ur, xlim=(-1.2, NCOLS + 1.2), ylim=(-1.2, NROWS + 1.2))

    # -------------------------------------------------------------------------
    # Bottom right: interpolate on overlap-expanded chunks
    # -------------------------------------------------------------------------
    _draw_pixel_grid(ax_br, nrows=NROWS, ncols=NCOLS, extend=1.0)
    _draw_chunk_boundaries(ax_br, extend=1.0)
    _draw_overlap_chunk(ax_br, overlap_bounds)
    _draw_highlight_chunk(ax_br, chunk_bounds, color=HIGHLIGHT_COLOR, lw=EXAMPLE_CHUNK_LW)
    _draw_points_on_raster(ax_br, highlight_ids=POINT_IDS_HIGHLIGHT, show_ids=False)

    ax_br.text(
        0.5,
        1.02,
        "Interpolate on overlapped chunks",
        transform=ax_br.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=NEUTRAL,
    )
    ax_br.text(
        0.5,
        -0.05,
        f"Loop on expanded raster chunks to interpolate\n"
        f"(overlap size depends on resampling method)",
        transform=ax_br.transAxes,
        ha="center",
        va="top",
        fontsize=10.2,
        color=NEUTRAL,
        linespacing=1.25,
    )
    _setup_axis(ax_br, xlim=(-1.2, NCOLS + 1.2), ylim=(-1.2, NROWS + 1.2))

    # -------------------------------------------------------------------------
    # Bottom left: reordered outputs
    # -------------------------------------------------------------------------
    _draw_point_ribbon(
        ax_bl,
        highlight_chunk=HIGHLIGHT_POINT_CHUNK,
        values=OUTPUT_VALUES,
        title="Reordered output values",
    )
    ax_bl.text(
        0.5,
        1.02,
        "Reorder outputs",
        transform=ax_bl.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=NEUTRAL,
    )
    ax_bl.text(
        0.5,
        0,
        "Per-block interpolation results are concatenated and\nreordered to the original point sequence",
        transform=ax_bl.transAxes,
        ha="center",
        va="top",
        fontsize=10.2,
        color=NEUTRAL,
        linespacing=1.25,
    )
    _setup_axis(ax_bl, xlim=(-0.4, len(POINT_X) + 0.4), ylim=(-1.0, 1.4))
    ax_bl.set_aspect("auto")

    # -------------------------------------------------------------------------
    # Global title
    # -------------------------------------------------------------------------
    fig.text(
        0.5,
        0.99,
        "Chunked raster interpolation at points",
        ha="center",
        va="top",
        fontsize=15,
        fontweight="semibold",
        color="0.35",
    )

    # -------------------------------------------------------------------------
    # Workflow arrows
    # -------------------------------------------------------------------------
    fig.canvas.draw()

    pos_ul = ax_ul.get_position()
    pos_ur = ax_ur.get_position()
    pos_br = ax_br.get_position()
    pos_bl = ax_bl.get_position()

    # UL to UR
    _add_workflow_arrow(
        fig,
        (pos_ul.x1 + 0.01, 0.5 * (pos_ul.y0 + pos_ul.y1)),
        (pos_ur.x0 - 0.01, 0.5 * (pos_ur.y0 + pos_ur.y1)),
        rad=0.0,
    )

    # UR to BR (curved outward to avoid the BR title)
    _add_workflow_arrow(
        fig,
        (pos_ur.x1 - 0.25, 0.5 * (pos_ur.y0 + pos_ur.y1) - 0.02),
        (pos_br.x1 - 0.25, 0.5 * (pos_br.y0 + pos_br.y1) + 0.02),
        rad=0.45,
    )

    # BR to BL
    _add_workflow_arrow(
        fig,
        (pos_br.x0 - 0.01, 0.5 * (pos_br.y0 + pos_br.y1)),
        (pos_bl.x1 + 0.01, 0.5 * (pos_bl.y0 + pos_bl.y1)),
        rad=0.0,
    )

    _add_left_column_separator(fig, ax_ul, ax_bl)
    _add_visual_legend(fig)

    fig.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.16)
    return fig, axes


fig, _ = make_chunked_interp_points_diagram()
plt.show()