"""Script to make diagram for chunked polygonize in documentation."""
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Rectangle

from geoutils.multiproc.chunked import ChunkedGeoGrid, GeoGrid
from geoutils.interface.vectorization import (
    _chunked_build_dst_geotiling,
    _chunked_clip_gdf_to_bounds_polygonal,
    _chunked_label_block_per_value,
    _chunked_polygonize_block_labels,
    _chunked_seam_pairs_from_strips,
    _polygonize_base,
)


# -----------------------------------------------------------------------------
# Example raster data and chunk layout
# -----------------------------------------------------------------------------

ARR = np.array(
    [
        [1, 1, 1, 1, 1, 0],
        [1, 1, 0, 0, 1, 0],
        [0, 0, 2, 2, 2, 0],
        [0, 3, 3, 3, 0, 0],
        [0, 3, 0, 3, 0, 0],
    ],
    dtype=np.uint8,
)

NROWS, NCOLS = ARR.shape
SPLIT_COL = 3
HALO = 1
CONNECTIVITY = 4
FLOAT_TOL = 0.001

VALUE_COLORS: dict[int, str] = {
    0: "#FFFFFF",
    1: "#8ECAE6",
    2: "#B7E4C7",
    3: "#FFD166",
}

NEUTRAL = "#333333"
PIXEL_GRID = "0.80"
CHUNK_COLOR = "#222222"
STITCH_COLOR = "#F58518"
POLYGON_COLOR = "#555555"


# -----------------------------------------------------------------------------
# Small containers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class BlockData:
    """Per-block arrays and metadata used for the schematic."""
    block_id: int
    block_id_dict: dict[str, int]
    geogrid: GeoGrid
    values: np.ndarray
    mask: np.ndarray
    labels: np.ndarray
    gdf_labels: gpd.GeoDataFrame
    gdf_geom: gpd.GeoDataFrame


# -----------------------------------------------------------------------------
# Coordinate helpers
# -----------------------------------------------------------------------------

def cell_xy(i: int, j: int, nrows: int) -> tuple[float, float]:
    """Bottom-left of a raster cell in plotting coordinates."""
    return float(j), float(nrows - 1 - i)


def cell_center(i: int, j: int, nrows: int) -> tuple[float, float]:
    """Center of a raster cell in plotting coordinates."""
    x, y = cell_xy(i, j, nrows)
    return x + 0.5, y + 0.5


def centroid_from_cells(cells: list[tuple[int, int]], nrows: int) -> tuple[float, float]:
    """Centroid from a list of global raster cells."""
    xs = []
    ys = []
    for i, j in cells:
        xc, yc = cell_center(i, j, nrows)
        xs.append(xc)
        ys.append(yc)
    return float(np.mean(xs)), float(np.mean(ys))


# -----------------------------------------------------------------------------
# Build block data using our package functions
# -----------------------------------------------------------------------------

def build_demo_blocks() -> tuple[ChunkedGeoGrid, list[GeoGrid], list[dict[str, int]], list[BlockData]]:
    """Build a 2-chunk horizontal tiling and derive block products with package helpers."""
    transform = rio.transform.from_origin(0.0, float(NROWS), 1.0, 1.0)
    crs = 4326
    chunks = ((NROWS,), (SPLIT_COL, NCOLS - SPLIT_COL))

    tiling, block_geogrids, block_ids = _chunked_build_dst_geotiling(
        shape=ARR.shape,
        transform=transform,
        crs=crs,
        chunks=chunks,
    )

    blocks: list[BlockData] = []

    for block_id, (b, gg) in enumerate(zip(block_ids, block_geogrids, strict=True)):
        ys, ye, xs, xe = b["ys"], b["ye"], b["xs"], b["xe"]
        values = ARR[ys:ye, xs:xe]
        mask = values != 0

        labels = _chunked_label_block_per_value(
            values,
            mask,
            connectivity=CONNECTIVITY,
            float_tol=FLOAT_TOL,
        )

        gdf_labels = _chunked_polygonize_block_labels(
            labels=labels,
            values=values,
            mask=mask,
            transform=gg.transform,
            value_column="raster_value",
            connectivity=CONNECTIVITY,
            local_id_column="local_id",
            float_tol=FLOAT_TOL,
        )

        ys_h = max(0, ys - HALO)
        ye_h = min(NROWS, ye + HALO)
        xs_h = max(0, xs - HALO)
        xe_h = min(NCOLS, xe + HALO)

        values_h = ARR[ys_h:ye_h, xs_h:xe_h]
        mask_h = values_h != 0
        transform_h = transform * rio.Affine.translation(xs_h, ys_h)

        gdf_geom = _polygonize_base(
            values_h,
            mask_h,
            transform=transform_h,
            crs=crs,
            data_column_name="polygon_id",
            value_column="raster_value",
            connectivity=CONNECTIVITY,
            float_tol=FLOAT_TOL,
        )

        bb_interior = gg.bounds
        gdf_geom = _chunked_clip_gdf_to_bounds_polygonal(
            gdf_geom,
            bb_interior,
            keep_border=True,
            area_eps=0.0,
        )

        blocks.append(
            BlockData(
                block_id=block_id,
                block_id_dict=b,
                geogrid=gg,
                values=values,
                mask=mask,
                labels=labels,
                gdf_labels=gdf_labels,
                gdf_geom=gdf_geom,
            )
        )

    return tiling, block_geogrids, block_ids, blocks


def label_cells_global(labels: np.ndarray, block: BlockData, local_id: int) -> list[tuple[int, int]]:
    """Global raster cells belonging to one local component label."""
    ys0 = block.block_id_dict["ys"]
    xs0 = block.block_id_dict["xs"]
    ii, jj = np.where(labels == local_id)
    return [(ys0 + int(i), xs0 + int(j)) for i, j in zip(ii, jj, strict=True)]


def build_seam_pairs(left: BlockData, right: BlockData) -> list[tuple[int, int, int, int]]:
    """Use the package seam helper on the vertical seam between the two blocks."""
    left_lab = left.labels[:, -1:]
    right_lab = right.labels[:, :1]
    left_val = left.values[:, -1:]
    right_val = right.values[:, :1]
    left_mask = left.mask[:, -1:]
    right_mask = right.mask[:, :1]

    pairs64 = _chunked_seam_pairs_from_strips(
        left_lab,
        right_lab,
        left_val,
        right_val,
        left_mask,
        right_mask,
        left_block_id=left.block_id,
        right_block_id=right.block_id,
        connectivity=CONNECTIVITY,
        axis="v",
        float_tol=FLOAT_TOL,
    )

    out: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()

    for a, b in pairs64:
        left_label = int(np.int64(a) & 0xFFFFFFFF)
        right_label = int(np.int64(b) & 0xFFFFFFFF)

        seam_rows = np.where(
            (left.labels[:, -1] == left_label)
            & (right.labels[:, 0] == right_label)
            & (left.values[:, -1] == right.values[:, 0])
            & left.mask[:, -1]
            & right.mask[:, 0]
        )[0]
        if seam_rows.size == 0:
            continue

        row = int(seam_rows[0])
        value = int(left.values[row, -1])
        key = (value, left_label, right_label)
        if key not in seen:
            seen.add(key)
            out.append((value, left_label, right_label, row))

    return out


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def draw_cells(ax: plt.Axes, arr: np.ndarray, *, alpha: float = 1.0) -> None:
    """Draw colored raster cells."""
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            x, y = cell_xy(i, j, arr.shape[0])
            rect = Rectangle(
                (x, y),
                1.0,
                1.0,
                facecolor=VALUE_COLORS[int(arr[i, j])],
                edgecolor="none",
                linewidth=0.0,
                alpha=alpha,
                zorder=1,
            )
            ax.add_patch(rect)


def draw_values(ax: plt.Axes, arr: np.ndarray) -> None:
    """Draw raster values in cell centers."""
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = int(arr[i, j])
            if val == 0:
                continue
            xc, yc = cell_center(i, j, arr.shape[0])
            ax.text(
                xc,
                yc,
                str(val),
                ha="center",
                va="center",
                fontsize=11.5,
                color=NEUTRAL,
                zorder=5,
            )


def draw_pixel_grid(ax: plt.Axes) -> None:
    """Thin pixel grid for the toy raster."""
    segments: list[np.ndarray] = []
    for x in range(NCOLS + 1):
        segments.append(np.array([[x, 0], [x, NROWS]], dtype=float))
    for y in range(NROWS + 1):
        segments.append(np.array([[0, y], [NCOLS, y]], dtype=float))

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


def draw_chunk_boundaries(ax: plt.Axes, geogrids: list[GeoGrid]) -> None:
    """Draw only the internal chunk boundary (seam), not the outer perimeter."""
    ax.plot(
        [SPLIT_COL, SPLIT_COL],
        [0, NROWS],
        color=CHUNK_COLOR,
        linewidth=2.5,
        zorder=6,
        solid_capstyle="round",
    )


def draw_gdf_boundaries(
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    *,
    color: str,
    linewidth: float,
    zorder: int,
) -> None:
    """Plot GeoDataFrame boundaries as linework."""
    if len(gdf) == 0:
        return

    try:
        boundary = gdf.geometry.union_all().boundary
    except Exception:
        boundary = gdf.geometry.unary_union.boundary

    geoms = getattr(boundary, "geoms", [boundary])
    segments: list[np.ndarray] = []

    for geom in geoms:
        if geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            segments.append(np.asarray(geom.coords))
        elif geom.geom_type == "MultiLineString":
            for sub in geom.geoms:
                segments.append(np.asarray(sub.coords))

    if not segments:
        return

    ax.add_collection(
        LineCollection(
            segments,
            colors=color,
            linewidths=linewidth,
            capstyle="round",
            joinstyle="round",
            zorder=zorder,
        )
    )


def draw_local_labels(ax: plt.Axes, block: BlockData, *, prefix: str) -> None:
    """Draw local label ids from actual per-block label rasters."""
    local_ids = sorted(int(v) for v in np.unique(block.labels) if v > 0)
    for lid in local_ids:
        cells = label_cells_global(block.labels, block, lid)
        xc, yc = centroid_from_cells(cells, NROWS)
        ax.text(
            xc,
            yc,
            f"{prefix}{lid}",
            ha="center",
            va="center",
            fontsize=9.5,
            color=NEUTRAL,
            bbox=dict(
                boxstyle="round,pad=0.18",
                fc="white",
                ec=NEUTRAL,
                lw=0.8,
                alpha=0.95,
            ),
            zorder=12,
        )


def draw_union_links(ax: plt.Axes, seam_pairs: list[tuple[int, int, int, int]], left: BlockData, right: BlockData) -> None:
    """Draw seam equivalence links for label_union."""
    for gid, (value, ll, rr, row) in enumerate(seam_pairs, start=1):
        left_cells = label_cells_global(left.labels, left, ll)
        right_cells = label_cells_global(right.labels, right, rr)

        x1, y1 = centroid_from_cells(left_cells, NROWS)
        x2, y2 = centroid_from_cells(right_cells, NROWS)

        conn = ConnectionPatch(
            (x1 + 0.35, y1),
            (x2 - 0.35, y2),
            coordsA="data",
            coordsB="data",
            axesA=ax,
            axesB=ax,
            arrowstyle="<->",
            color=NEUTRAL,
            linewidth=1.5,
            mutation_scale=11,
            zorder=11,
        )
        ax.add_artist(conn)

        ym = NROWS - 1 - row + 0.5
        ax.text(
            SPLIT_COL,
            ym,
            f"G{gid}",
            ha="center",
            va="center",
            fontsize=8.8,
            color=NEUTRAL,
            bbox=dict(
                boxstyle="round,pad=0.15",
                fc=VALUE_COLORS[value],
                ec=NEUTRAL,
                lw=0.7,
                alpha=0.95,
            ),
            zorder=13,
        )


def draw_stitch_links(ax: plt.Axes, seam_pairs: list[tuple[int, int, int, int]], left: BlockData, right: BlockData) -> None:
    """Draw vector-stitch links across the seam."""
    for _, ll, rr, _row in seam_pairs:
        left_rows = np.where(left.labels[:, -1] == ll)[0]
        right_rows = np.where(right.labels[:, 0] == rr)[0]
        if left_rows.size == 0 or right_rows.size == 0:
            continue

        li = int(left_rows[0])
        ri = int(right_rows[0])

        x1, y1 = cell_center(li, SPLIT_COL - 1, NROWS)
        x2, y2 = cell_center(ri, SPLIT_COL, NROWS)

        conn = ConnectionPatch(
            (x1 + 0.35, y1),
            (x2 - 0.35, y2),
            coordsA="data",
            coordsB="data",
            axesA=ax,
            axesB=ax,
            arrowstyle="-",
            connectionstyle="arc3,rad=0.25",
            color=STITCH_COLOR,
            linewidth=2.4,
            alpha=0.95,
            zorder=12,
        )
        ax.add_artist(conn)


def draw_halo_seam(ax: plt.Axes) -> None:
    """Draw halo region along the chunk boundary, extending slightly outside the raster."""
    rect = Rectangle(
        (SPLIT_COL - HALO, -0.1),
        2 * HALO,
        NROWS + 0.2,
        facecolor=STITCH_COLOR,
        edgecolor="none",
        alpha=0.4,
        zorder=3,
        clip_on=False,
    )
    ax.add_patch(rect)


def setup_axis(ax: plt.Axes, *, halo_pad: bool = False) -> None:
    """Common axis formatting."""
    extra = 1.1 if halo_pad else 0.0
    ax.set_xlim(-extra, NCOLS + extra)
    ax.set_ylim(-extra, NROWS + extra)
    ax.set_aspect("equal")

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)


def _add_visual_legend(ax: plt.Axes) -> None:
    """
    Draw a compact multi-row visual legend inside the bottom-right of an axis.
    """

    # Anchor in axis coordinates
    x0 = 1.6
    y0 = 0.74
    dy = 0.15
    line_len = 0.10
    text_dx = 0.15

    text_kw = dict(
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=11,
        color=NEUTRAL,
        clip_on=False,
    )

    # Row 1: pixel grid
    y = y0
    ax.plot(
        [x0, x0 + line_len],
        [y, y],
        transform=ax.transAxes,
        color=PIXEL_GRID,
        lw=0.8,
        solid_capstyle="round",
        clip_on=False,
        zorder=20,
    )
    ax.text(x0 + text_dx, y, "Pixel grid", **text_kw)

    # Row 2: chunk boundary
    y = y0 - dy
    ax.plot(
        [x0, x0 + line_len],
        [y, y],
        transform=ax.transAxes,
        color=CHUNK_COLOR,
        lw=2.2,
        solid_capstyle="round",
        clip_on=False,
        zorder=20,
    )
    ax.text(x0 + text_dx, y, "Chunk boundary", **text_kw)

    # Row 3: polygon outlines
    y = y0 - 2 * dy
    ax.plot(
        [x0, x0 + line_len],
        [y, y],
        transform=ax.transAxes,
        color=POLYGON_COLOR,
        lw=1.6,
        solid_capstyle="round",
        clip_on=False,
        zorder=20,
    )
    ax.text(x0 + text_dx, y, "Polygon outlines", **text_kw)

    # Row 4: vector stitch
    y = y0 - 3 * dy
    arrow = FancyArrowPatch(
        (x0, y),
        (x0 + line_len, y),
        transform=ax.transAxes,
        arrowstyle="-",
        connectionstyle="arc3,rad=0.4",
        lw=2.4,
        color=STITCH_COLOR,
        clip_on=False,
        zorder=20,
    )
    ax.add_artist(arrow)
    ax.text(x0 + text_dx, y, "Vector stitch", **text_kw)

    # Row 5: halo window
    y = y0 - 4 * dy
    rect = Rectangle(
        (x0, y - 0.025),
        line_len,
        0.05,
        transform=ax.transAxes,
        facecolor=STITCH_COLOR,
        edgecolor="none",
        alpha=0.4,
        clip_on=False,
        zorder=20,
    )
    ax.add_patch(rect)
    ax.text(x0 + text_dx, y, "Halo window", **text_kw)

def _add_polygonize_title(fig: plt.Figure, strategy_axes: list[plt.Axes]) -> None:
    """Add grouped title and underline above the three strategy panels."""
    pos_left = strategy_axes[0].get_position()
    pos_right = strategy_axes[-1].get_position()

    x_center = 0.5 * (pos_left.x0 + pos_right.x1)
    y_text = pos_left.y1 + 0.14
    y_line = pos_left.y1 + 0.137

    fig.text(
        x_center,
        y_text,
        "Chunked polygonization strategies",
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


# -----------------------------------------------------------------------------
# Main figure
# -----------------------------------------------------------------------------

def make_chunked_polygonize_diagram() -> tuple[plt.Figure, np.ndarray]:
    """Build a 2-row schematic diagram for the chunked polygonize strategies."""
    _tiling, block_geogrids, _block_ids, blocks = build_demo_blocks()
    left, right = blocks
    seam_pairs = build_seam_pairs(left, right)

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 1.35],
        hspace=0.32,
        wspace=0.18,
    )

    ax_input = fig.add_subplot(gs[0, 1])
    ax_union = fig.add_subplot(gs[1, 0])
    ax_stitch = fig.add_subplot(gs[1, 1])
    ax_geom = fig.add_subplot(gs[1, 2])

    axes = np.array([ax_input, ax_union, ax_stitch, ax_geom], dtype=object)

    titles = ["Input raster", "label_union", "label_stitch", "geometry_stitch"]
    subtitles = [
        "Zoom on raster pixels\nat the vertical boundary\nbetween two chunks",
        "1. Label connected components per chunk\n2. Merge labels across chunk boundaries\n3. Polygonize merged "
        "labels and dissolve",
        "1. Label connected components per chunk\n2. Polygonize labels per chunk\n3. Stitch polygons across chunk boundaries",
        "1. Polygonize halo-expanded chunks\n2. Clip polygons to chunk interior\n3. Stitch polygons across chunk boundaries",
    ]

    # Top centered panel: input
    draw_cells(ax_input, ARR)
    draw_pixel_grid(ax_input)
    draw_chunk_boundaries(ax_input, block_geogrids)
    draw_values(ax_input, ARR)
    ax_input.text(
        0.5,
        1.02,
        titles[0],
        transform=ax_input.transAxes,
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        color=NEUTRAL,
    )
    ax_input.text(
        -0.5,
        0.5,
        subtitles[0],
        transform=ax_input.transAxes,
        ha="center",
        va="center",
        fontsize=10.5,
        color=NEUTRAL,
        linespacing=1.25,
    )
    setup_axis(ax_input)

    # Bottom row: label_union
    draw_cells(ax_union, ARR, alpha=0.92)
    draw_pixel_grid(ax_union)
    draw_chunk_boundaries(ax_union, block_geogrids)
    draw_local_labels(ax_union, left, prefix="L")
    draw_local_labels(ax_union, right, prefix="R")
    draw_union_links(ax_union, seam_pairs, left, right)
    ax_union.text(
        0.5,
        1.02,
        titles[1],
        transform=ax_union.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=NEUTRAL,
    )
    ax_union.text(
        0.5,
        -0.1,
        subtitles[1],
        transform=ax_union.transAxes,
        ha="center",
        va="top",
        fontsize=10.2,
        color=NEUTRAL,
        linespacing=1.25,
    )
    setup_axis(ax_union)

    # Bottom row: label_stitch
    draw_cells(ax_stitch, ARR, alpha=0.92)
    draw_pixel_grid(ax_stitch)
    draw_chunk_boundaries(ax_stitch, block_geogrids)
    draw_gdf_boundaries(ax_stitch, left.gdf_geom, color=POLYGON_COLOR, linewidth=1.4, zorder=9)
    draw_gdf_boundaries(ax_stitch, right.gdf_geom, color=POLYGON_COLOR, linewidth=1.4, zorder=9)
    draw_stitch_links(ax_stitch, seam_pairs, left, right)
    ax_stitch.text(
        0.5,
        1.02,
        titles[2],
        transform=ax_stitch.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=NEUTRAL,
    )
    ax_stitch.text(
        0.5,
        -0.1,
        subtitles[2],
        transform=ax_stitch.transAxes,
        ha="center",
        va="top",
        fontsize=10.2,
        color=NEUTRAL,
        linespacing=1.25,
    )
    setup_axis(ax_stitch)

    # Bottom row: geometry_stitch
    draw_cells(ax_geom, ARR, alpha=0.92)
    draw_pixel_grid(ax_geom)
    draw_chunk_boundaries(ax_geom, block_geogrids)
    draw_halo_seam(ax_geom)
    draw_gdf_boundaries(ax_geom, left.gdf_geom, color=POLYGON_COLOR, linewidth=1.4, zorder=9)
    draw_gdf_boundaries(ax_geom, right.gdf_geom, color=POLYGON_COLOR, linewidth=1.4, zorder=9)
    draw_stitch_links(ax_geom, seam_pairs, left, right)
    ax_geom.text(
        0.5,
        1.02,
        titles[3],
        transform=ax_geom.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=NEUTRAL,
    )
    ax_geom.text(
        0.5,
        -0.1,
        subtitles[3],
        transform=ax_geom.transAxes,
        ha="center",
        va="top",
        fontsize=10.2,
        color=NEUTRAL,
        linespacing=1.25,
    )
    setup_axis(ax_geom)

    _add_visual_legend(ax_input)
    _add_polygonize_title(fig, [ax_union, ax_stitch, ax_geom])

    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.15)

    return fig, axes


fig, _ = make_chunked_polygonize_diagram()
plt.show()
