"""Script to make diagram for chunked rasterize in documentation."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Polygon as MplPolygon, Rectangle, ConnectionPatch
from shapely.geometry import Polygon, box

from geoutils.interface.rasterization import _rasterio_rasterize_burn

# -----------------------------------------------------------------------------
# Example raster grid, chunks, and vector geometries
# -----------------------------------------------------------------------------

NROWS = 8
NCOLS = 10

CHUNK_ROWS = (4, 4)
CHUNK_COLS = (5, 5)

# Highlight bottom-right chunk
HIGHLIGHT_CHUNK = (1, 1)

NEUTRAL = "#333333"
PIXEL_GRID = "0.80"
CHUNK_COLOR = "#222222"
HIGHLIGHT_COLOR = "#F58518"
POLYGON_COLOR = "#6BAED6"
POLYGON_EDGE = "#4C78A8"
FADED_ALPHA = 0.22

# Three example polygons in raster coordinates
POLYGONS = [
    Polygon([(0.8, 6.8), (3.2, 7.4), (4.1, 5.7), (2.4, 4.8), (0.9, 5.6)]),
    Polygon([(4.2, 5.9), (7.4, 6.7), (8.2, 4.5), (6.7, 3.2), (4.6, 4.0)]),
    Polygon([(6.0, 2.5), (8.7, 2.9), (9.1, 0.8), (6.8, 0.6), (5.7, 1.4)]),
]
POLY_VALUES = np.array([1, 2, 3], dtype=np.uint8)


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def _data_point_to_fig(fig: plt.Figure, ax: plt.Axes, x_data: float, y_data: float) -> tuple[float, float]:
    """Convert a data-coordinate point to figure coordinates."""
    xy_disp = ax.transData.transform((x_data, y_data))
    xy_fig = fig.transFigure.inverted().transform(xy_disp)
    return float(xy_fig[0]), float(xy_fig[1])

def chunk_bounds(chunk_location: tuple[int, int]) -> tuple[float, float, float, float]:
    """Return chunk bounds as (left, bottom, right, top)."""
    iy, ix = chunk_location

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


def polygon_to_patch(poly: Polygon, **kwargs) -> MplPolygon:
    """Convert shapely polygon to matplotlib patch."""
    return MplPolygon(np.asarray(poly.exterior.coords), closed=True, **kwargs)


def geometry_intersects_chunk(poly: Polygon, bounds: tuple[float, float, float, float]) -> bool:
    """Return whether polygon intersects chunk bbox."""
    left, bottom, right, top = bounds
    qbox = box(left, bottom, right, top)
    return poly.intersects(qbox)


def build_rasterized_chunk(
    bounds: tuple[float, float, float, float],
    polygons: list[Polygon],
    values: np.ndarray,
) -> np.ndarray:
    """Rasterize candidate polygons into the highlighted chunk using the package helper."""
    left, bottom, right, top = bounds
    width = int(right - left)
    height = int(top - bottom)

    transform = rio.transform.from_bounds(
        west=left,
        south=bottom,
        east=right,
        north=top,
        width=width,
        height=height,
    )

    geoms = np.asarray(polygons, dtype=object)

    return _rasterio_rasterize_burn(
        geoms=geoms,
        values=values,
        default_value=None,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def draw_pixel_grid(ax: plt.Axes, *, nrows: int, ncols: int, extend: float = 1.0) -> None:
    """Draw thin pixel grid, extended slightly outside the plotted raster."""
    segments: list[np.ndarray] = []

    # Vertical pixel lines
    for x in range(ncols + 1):
        segments.append(np.array([[x, -extend], [x, nrows + extend]], dtype=float))

    # Horizontal pixel lines
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

def draw_chunk_boundaries(ax: plt.Axes, *, extend: float = 1.0) -> None:
    """Draw all chunk boundaries, extended slightly outside the plotted raster."""
    row_starts = np.concatenate([[0], np.cumsum(CHUNK_ROWS)])
    col_starts = np.concatenate([[0], np.cumsum(CHUNK_COLS)])

    # Draw all vertical chunk boundaries, including outside ones
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

    # Draw all horizontal chunk boundaries, including outside ones
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

def draw_highlight_chunk(ax: plt.Axes, bounds: tuple[float, float, float, float]) -> None:
    """Draw highlighted chunk bbox."""
    left, bottom, right, top = bounds
    rect = Rectangle(
        (left, bottom),
        right - left,
        top - bottom,
        fill=False,
        edgecolor=HIGHLIGHT_COLOR,
        linewidth=2.8,
        zorder=5,
    )
    ax.add_patch(rect)

def draw_polygon_ids(
    ax: plt.Axes,
    polygons: list[Polygon],
    ids: list[int],
    *,
    candidate_mask: list[bool] | None = None,
) -> None:
    """Draw polygon IDs at their centroid."""
    for i, poly in enumerate(polygons):

        is_candidate = True if candidate_mask is None else candidate_mask[i]

        cx, cy = poly.centroid.coords[0]

        ax.text(
            cx,
            cy,
            f"ID: {ids[i]}",
            ha="center",
            va="center",
            fontsize=9,
            color=NEUTRAL,
            zorder=20,
            alpha=1.0 if is_candidate else 0.7,
            bbox=dict(
                boxstyle="round,pad=0.2",
                fc="white",
                ec="none",
                alpha=0.8,
            ),
        )

def draw_chunk_query_box(ax: plt.Axes, bounds: tuple[float, float, float, float]) -> None:
    """Draw filled highlight for candidate-query chunk."""
    left, bottom, right, top = bounds

    rect_fill = Rectangle(
        (left, bottom),
        right - left,
        top - bottom,
        facecolor=HIGHLIGHT_COLOR,
        edgecolor="none",
        alpha=0.10,
        zorder=2,
    )
    rect_edge = Rectangle(
        (left, bottom),
        right - left,
        top - bottom,
        fill=False,
        edgecolor=HIGHLIGHT_COLOR,
        linewidth=2.8,
        zorder=5,
    )
    ax.add_patch(rect_fill)
    ax.add_patch(rect_edge)


def draw_polygons(
    ax: plt.Axes,
    polygons: list[Polygon],
    *,
    candidate_mask: list[bool] | None = None,
) -> None:
    """Draw input polygons, optionally fading non-candidates."""
    for i, poly in enumerate(polygons):
        is_candidate = True if candidate_mask is None else candidate_mask[i]
        alpha = 0.45 if is_candidate else FADED_ALPHA
        edgecolor = POLYGON_EDGE if is_candidate else "#7A8FA6"
        facecolor = POLYGON_COLOR if is_candidate else "#BFD7EA"

        patch = polygon_to_patch(
            poly,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.5,
            alpha=alpha,
            zorder=4 if is_candidate else 2,
        )
        ax.add_patch(patch)


def draw_chunk_result(ax: plt.Axes, arr: np.ndarray) -> None:
    """Draw rasterized chunk result as a small chunk-local raster."""
    nrows, ncols = arr.shape

    value_colors = {
        0: "#FFFFFF",
        1: "#6BAED6",
        2: "#9ECAE1",
        3: "#C6DBEF",
    }

    for i in range(nrows):
        for j in range(ncols):
            val = int(arr[i, j])
            y = nrows - 1 - i
            x = j
            rect = Rectangle(
                (x, y),
                1.0,
                1.0,
                facecolor=value_colors[val],
                edgecolor="none",
                linewidth=0.0,
                zorder=1,
            )
            ax.add_patch(rect)

            if val != 0:
                ax.text(
                    x + 0.5,
                    y + 0.5,
                    str(val),
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=NEUTRAL,
                    zorder=4,
                )

    segments: list[np.ndarray] = []
    for x in range(ncols + 1):
        segments.append(np.array([[x, 0], [x, nrows]], dtype=float))
    for y in range(nrows + 1):
        segments.append(np.array([[0, y], [ncols, y]], dtype=float))

    ax.add_collection(
        LineCollection(
            segments,
            colors=PIXEL_GRID,
            linewidths=0.8,
            capstyle="round",
            joinstyle="round",
            zorder=2,
        )
    )

    rect = Rectangle(
        (0, 0),
        ncols,
        nrows,
        fill=False,
        edgecolor=HIGHLIGHT_COLOR,
        linewidth=2.8,
        zorder=5,
    )
    ax.add_patch(rect)


def setup_axis(ax: plt.Axes, *, nrows: int, ncols: int, pad: float = 0.6) -> None:
    """Common axis formatting."""
    ax.set_xlim(-pad, ncols + pad)
    ax.set_ylim(-pad, nrows + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_mapblocks_title(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    """Add grouped title over panels 2–3."""
    pos_left = axes[0].get_position()
    pos_right = axes[-1].get_position()

    x_center = 0.5 * (pos_left.x0 + pos_right.x1)
    y_text = pos_left.y1 + 0.195
    y_line = pos_left.y1 + 0.182

    fig.text(
        x_center,
        y_text,
        "Chunked rasterization",
        ha="center",
        va="bottom",
        fontsize=15,
        fontweight="semibold",
        color="0.35",
    )

    line = Line2D(
        [pos_left.x0, pos_right.x1],
        [y_line, y_line],
        transform=fig.transFigure,
        color="0.6",
        lw=1.0,
        alpha=0.9,
        solid_capstyle="round",
        zorder=100,
    )
    fig.add_artist(line)


def add_visual_legend(fig: plt.Figure) -> None:
    """Draw horizontal bottom legend."""
    y = 0.06
    x = 0.10
    dx = 0.18

    # Pixel grid
    fig.add_artist(
        Line2D(
            [x, x + 0.03],
            [y, y],
            transform=fig.transFigure,
            color=PIXEL_GRID,
            lw=0.8,
            solid_capstyle="round",
        )
    )
    fig.text(x + 0.038, y, "Pixel grid", transform=fig.transFigure, va="center", fontsize=10)

    # Chunk boundary
    x += dx
    fig.add_artist(
        Line2D(
            [x, x + 0.03],
            [y, y],
            transform=fig.transFigure,
            color=CHUNK_COLOR,
            lw=2.3,
        )
    )
    fig.text(x + 0.038, y, "Chunk grid", transform=fig.transFigure, va="center", fontsize=10)

    # Queried chunk
    x += dx
    rect = Rectangle(
        (x, y - 0.010),
        0.03,
        0.020,
        transform=fig.transFigure,
        facecolor=HIGHLIGHT_COLOR,
        edgecolor=HIGHLIGHT_COLOR,
        linewidth=1.4,
        alpha=0.15,
    )
    fig.add_artist(rect)
    fig.text(x + 0.038, y, "Queried chunk", transform=fig.transFigure, va="center", fontsize=10)

    # Candidate geometry
    x += dx
    poly = MplPolygon(
        np.array(
            [
                [x, y - 0.010],
                [x + 0.010, y + 0.012],
                [x + 0.022, y + 0.010],
                [x + 0.028, y - 0.008],
                [x + 0.012, y - 0.014],
            ]
        ),
        closed=True,
        transform=fig.transFigure,
        facecolor=POLYGON_COLOR,
        edgecolor=POLYGON_EDGE,
        linewidth=1.0,
        alpha=0.45,
    )
    fig.add_artist(poly)
    fig.text(x + 0.038, y, "Candidate geometry", transform=fig.transFigure, va="center", fontsize=10)

def chunk_center(bounds: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return center of chunk bounds."""
    left, bottom, right, top = bounds
    return 0.5 * (left + right), 0.5 * (bottom + top)


def add_between_panel_arrows(
    fig: plt.Figure,
    ax0: plt.Axes,
    ax1: plt.Axes,
    ax2: plt.Axes,
    highlight_bounds: tuple[float, float, float, float],
    chunk_result_shape: tuple[int, int],
) -> None:
    """Add arrows anchored to meaningful data coordinates in each panel."""
    c01 = chunk_center(highlight_bounds)
    c12 = chunk_center(highlight_bounds)

    # Center of chunk-local raster in panel 3
    nrows, ncols = chunk_result_shape
    c2 = (0.5 * ncols, 0.5 * nrows)

    # Panel 1 to Panel 2
    arr01 = ConnectionPatch(
        xyA=c01,
        coordsA=ax0.transData,
        xyB=c01,
        coordsB=ax1.transData,
        arrowstyle="-|>",
        linewidth=2.2,
        color=NEUTRAL,
        mutation_scale=22,
        shrinkA=8,
        shrinkB=8,
        connectionstyle="arc3,rad=0.0",
        zorder=30,
        clip_on=False,
    )
    fig.add_artist(arr01)

    # Panel 2 to Panel 3
    arr12 = ConnectionPatch(
        xyA=c12,
        coordsA=ax1.transData,
        xyB=c2,
        coordsB=ax2.transData,
        arrowstyle="-|>",
        linewidth=2.2,
        color=NEUTRAL,
        mutation_scale=22,
        shrinkA=8,
        shrinkB=8,
        connectionstyle="arc3,rad=0.0",
        zorder=30,
        clip_on=False,
    )
    fig.add_artist(arr12)

# -----------------------------------------------------------------------------
# Main figure
# -----------------------------------------------------------------------------

def make_chunked_rasterize_diagram() -> tuple[plt.Figure, np.ndarray]:
    """Build a 3-panel schematic for chunked rasterize."""
    highlight_bounds = chunk_bounds(HIGHLIGHT_CHUNK)
    candidate_mask = [geometry_intersects_chunk(poly, highlight_bounds) for poly in POLYGONS]
    candidate_polygons = [poly for poly, keep in zip(POLYGONS, candidate_mask, strict=True) if keep]
    candidate_values = POLY_VALUES[np.array(candidate_mask, dtype=bool)]
    chunk_result = build_rasterized_chunk(highlight_bounds, candidate_polygons, candidate_values)

    fig = plt.figure(figsize=(10, 4))
    gs = fig.add_gridspec(1, 3, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    axes = np.array([ax0, ax1, ax2], dtype=object)

    titles = [
        "Output raster grid",
        "Per-chunk geometry filtering",
        "Per-chunk rasterization",
    ]
    subtitles = [
        "Chunked output raster overlaid with\nvector geometries to burn",
        "Only geometries intersecting the highlighted\nchunk bounds are passed to this block",
        "Each chunk rasterizes only its candidate\ngeometries, or exits early if none intersect",
    ]

    # Panel 1
    draw_pixel_grid(ax0, nrows=NROWS, ncols=NCOLS)
    draw_chunk_boundaries(ax0)
    draw_polygons(ax0, POLYGONS)
    draw_polygon_ids(ax0, POLYGONS, POLY_VALUES.tolist())
    draw_highlight_chunk(ax0, highlight_bounds)
    ax0.text(
        0.5,
        1.05,
        titles[0],
        transform=ax0.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=NEUTRAL,
    )
    ax0.text(
        0.5,
        -0.07,
        subtitles[0],
        transform=ax0.transAxes,
        ha="center",
        va="top",
        fontsize=10.2,
        color=NEUTRAL,
        linespacing=1.25,
    )
    setup_axis(ax0, nrows=NROWS, ncols=NCOLS, pad=1.2)

    # Panel 2
    draw_pixel_grid(ax1, nrows=NROWS, ncols=NCOLS)
    draw_chunk_boundaries(ax1)
    draw_polygons(ax1, POLYGONS, candidate_mask=candidate_mask)
    draw_polygon_ids(ax1, POLYGONS, POLY_VALUES.tolist(), candidate_mask=candidate_mask)
    draw_chunk_query_box(ax1, highlight_bounds)
    ax1.text(
        0.5,
        1.05,
        titles[1],
        transform=ax1.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=NEUTRAL,
    )
    ax1.text(
        0.5,
        -0.07,
        subtitles[1],
        transform=ax1.transAxes,
        ha="center",
        va="top",
        fontsize=10.2,
        color=NEUTRAL,
        linespacing=1.25,
    )
    setup_axis(ax1, nrows=NROWS, ncols=NCOLS, pad=1.2)

    # Panel 3
    draw_chunk_result(ax2, chunk_result)
    ax2.text(
        0.5,
        1.05,
        titles[2],
        transform=ax2.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=NEUTRAL,
    )
    ax2.text(
        0.5,
        -0.07,
        subtitles[2],
        transform=ax2.transAxes,
        ha="center",
        va="top",
        fontsize=10.2,
        color=NEUTRAL,
        linespacing=1.25,
    )
    setup_axis(ax2, nrows=chunk_result.shape[0], ncols=chunk_result.shape[1], pad=0.6)


    add_mapblocks_title(fig, [ax0, ax2])
    add_visual_legend(fig)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.1)

    # Finalize transforms before placing arrows
    fig.canvas.draw()

    # Horizontal chunk boundary in panels 1 and 2
    y_chunk_boundary = NROWS - CHUNK_ROWS[0]

    # For panel 3, align with the middle of the chunk-local raster
    y_chunk_result = chunk_result.shape[0] / 2

    pos0 = ax0.get_position()
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()

    # Arrow 1: panel 1 to panel 2
    _, y0 = _data_point_to_fig(fig, ax0, NCOLS / 2, y_chunk_boundary)
    _, y1 = _data_point_to_fig(fig, ax1, NCOLS / 2, y_chunk_boundary)
    y01 = 0.5 * (y0 + y1)

    arrow01 = FancyArrowPatch(
        (pos0.x1 + 0.015, y01),
        (pos1.x0 - 0.015, y01),
        transform=fig.transFigure,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.0",
        linewidth=2.0,
        color=NEUTRAL,
        mutation_scale=20,
        shrinkA=0,
        shrinkB=0,
        zorder=30,
    )
    fig.add_artist(arrow01)

    # Arrow 2: panel 2 to panel 3
    _, y1b = _data_point_to_fig(fig, ax1, NCOLS / 2, y_chunk_boundary)
    _, y2 = _data_point_to_fig(fig, ax2, chunk_result.shape[1] / 2, y_chunk_result)
    y12 = 0.5 * (y1b + y2)

    arrow12 = FancyArrowPatch(
        (pos1.x1 + 0.015, y12),
        (pos2.x0 - 0.015, y12),
        transform=fig.transFigure,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=0.0",
        linewidth=2.0,
        color=NEUTRAL,
        mutation_scale=20,
        shrinkA=0,
        shrinkB=0,
        zorder=30,
    )
    fig.add_artist(arrow12)

    return fig, axes


fig, _ = make_chunked_rasterize_diagram()
plt.show()