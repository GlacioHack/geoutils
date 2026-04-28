"""Script to make diagram for chunked reproject in documentation."""
from __future__ import annotations

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import numpy as np
import rasterio as rio
from shapely.geometry import Polygon, MultiPolygon
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection
from shapely.geometry import LineString, MultiLineString
from geoutils.multiproc.chunked import GeoGrid, ChunkedGeoGrid, cached_cumsum


# -----------------------------------------------------------------------------
# Plotting utilities
# -----------------------------------------------------------------------------

def _geom_to_lines(geom) -> list[np.ndarray]:
    """Convert shapely line geometry to matplotlib line segments."""
    lines: list[np.ndarray] = []

    if geom.is_empty:
        return lines

    if isinstance(geom, LineString):
        lines.append(np.asarray(geom.coords))
    elif isinstance(geom, MultiLineString):
        for line in geom.geoms:
            lines.extend(_geom_to_lines(line))

    return lines


def _add_gdf_boundary(
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    *,
    color: str = "black",
    linewidth: float = 1.0,
    alpha: float = 1.0,
    zorder: int = 1,
) -> None:
    """Draw GeoDataFrame boundaries as linework, with each edge drawn only once."""
    boundary = gdf.geometry.boundary.union_all()
    lines = _geom_to_lines(boundary)

    if len(lines) == 0:
        return

    collection = LineCollection(
        lines,
        colors=color,
        linewidths=linewidth,
        alpha=alpha,
        zorder=zorder,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_collection(collection)


def _geom_to_patches(geom) -> list[MplPolygon]:
    """Convert shapely polygon / multipolygon to matplotlib patches."""
    patches: list[MplPolygon] = []

    if geom.is_empty:
        return patches

    if isinstance(geom, Polygon):
        patches.append(MplPolygon(np.asarray(geom.exterior.coords), closed=True))
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            patches.extend(_geom_to_patches(poly))
    else:
        pass

    return patches


def _add_gdf(
    ax: plt.Axes,
    gdf: gpd.GeoDataFrame,
    *,
    facecolor: str = "none",
    edgecolor: str = "black",
    linewidth: float = 1.0,
    alpha: float = 1.0,
    zorder: int = 1,
) -> None:
    """Add all geometries of a GeoDataFrame as matplotlib patches."""
    patches: list[MplPolygon] = []
    for geom in gdf.geometry:
        patches.extend(_geom_to_patches(geom))

    if len(patches) == 0:
        return

    collection = PatchCollection(
        patches,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        zorder=zorder,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_collection(collection)


def _set_ax_extent_from_gdfs(ax: plt.Axes, gdfs: list[gpd.GeoDataFrame], pad_frac: float = 0.08) -> None:
    """Set equal aspect and padded extent from one or more GeoDataFrames."""
    bounds = np.array([gdf.total_bounds for gdf in gdfs], dtype=float)
    xmin = np.min(bounds[:, 0])
    ymin = np.min(bounds[:, 1])
    xmax = np.max(bounds[:, 2])
    ymax = np.max(bounds[:, 3])

    dx = xmax - xmin
    dy = ymax - ymin
    padx = max(dx * pad_frac, 1e-12)
    pady = max(dy * pad_frac, 1e-12)

    ax.set_xlim(xmin - padx, xmax + padx)
    ax.set_ylim(ymin - pady, ymax + pady)
    ax.set_aspect("equal")
    ax.axis("off")


def _add_rectilinear_grid(
    ax: plt.Axes,
    grid: GeoGrid,
    *,
    color: str = "0.75",
    linewidth: float = 0.55,
    alpha: float = 0.8,
    zorder: int = 0,
) -> None:
    """Draw the internal pixel grid of a rectilinear GeoGrid."""
    left, bottom, right, top = grid.bounds
    dx, dy = grid.res

    xs = left + np.arange(grid.width + 1) * dx
    ys = top - np.arange(grid.height + 1) * dy

    segments: list[np.ndarray] = []

    for x in xs:
        segments.append(np.array([[x, bottom], [x, top]], dtype=float))

    for y in ys:
        segments.append(np.array([[left, y], [right, y]], dtype=float))

    collection = LineCollection(
        segments,
        colors=color,
        linewidths=linewidth,
        alpha=alpha,
        zorder=zorder,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_collection(collection)

def _add_chunk_size_arrows(
    ax: plt.Axes,
    chunk: GeoGrid,
    *,
    label_x: str,
    label_y: str,
    color: str = "#333333",
    fontsize: float = 8.5,
) -> None:
    """Add double-headed arrows showing the width/height of a chunk.

    The arrows are positioned using display coordinates so they look visually
    consistent even when x/y data units differ strongly.
    """
    left, bottom, right, top = chunk.bounds

    # Chunk corners in data coordinates
    p_bl = np.array([left, bottom])
    p_br = np.array([right, bottom])
    p_tl = np.array([left, top])

    # Transform to display coordinates
    to_disp = ax.transData.transform
    to_data = ax.transData.inverted().transform

    bl_d = to_disp(p_bl)
    br_d = to_disp(p_br)
    tl_d = to_disp(p_tl)

    # Visual chunk size in display units
    width_d = br_d[0] - bl_d[0]
    height_d = tl_d[1] - bl_d[1]

    # Offsets/padding in display units, so appearance is consistent
    ref_d = min(abs(width_d), abs(height_d))
    off_d = 0.25 * ref_d
    pad_d = 0.15 * ref_d

    h0_d = np.array([bl_d[0] - pad_d, bl_d[1] - off_d])
    h1_d = np.array([br_d[0] + pad_d, bl_d[1] - off_d])

    h0 = to_data(h0_d)
    h1 = to_data(h1_d)

    ax.add_patch(
        FancyArrowPatch(
            tuple(h0),
            tuple(h1),
            arrowstyle="<->",
            mutation_scale=11,
            linewidth=1.1,
            color=color,
            zorder=10,
            clip_on=False,
        )
    )

    htxt_d = np.array([0.5 * (h0_d[0] + h1_d[0]), bl_d[1] - 1.35 * off_d])
    htxt = to_data(htxt_d)
    ax.text(
        htxt[0],
        htxt[1],
        label_x,
        ha="center",
        va="top",
        fontsize=fontsize,
        color=color,
    )

    # Vertical arrow left of chunk (in display coordinates)
    v0_d = np.array([bl_d[0] - off_d, bl_d[1] - pad_d])
    v1_d = np.array([bl_d[0] - off_d, tl_d[1] + pad_d])

    v0 = to_data(v0_d)
    v1 = to_data(v1_d)

    ax.add_patch(
        FancyArrowPatch(
            tuple(v0),
            tuple(v1),
            arrowstyle="<->",
            mutation_scale=11,
            linewidth=1.1,
            color=color,
            zorder=10,
            clip_on=False,
        )
    )

    vtxt_d = np.array([bl_d[0] - 1.35 * off_d, 0.5 * (v0_d[1] + v1_d[1])])
    vtxt = to_data(vtxt_d)
    ax.text(
        vtxt[0],
        vtxt[1],
        label_y,
        ha="right",
        va="center",
        rotation=90,
        fontsize=fontsize,
        color=color,
    )

def _add_line_legend(
    ax: plt.Axes,
    *,
    thin_label: str = "Pixel grid",
    thick_label: str = "Chunk grid",
    thin_color: str = "0.75",
    thick_color: str = "#6BAED6",
    thick_color_2: str = "#F58518",
    dest_color: str = "#F58518",
) -> None:
    """Add a compact inline legend in axis coordinates."""

    xt = 0.65
    x0 = xt - 0.18
    x1 = xt - 0.12
    x2 = xt - 0.06

    y_top = -0.40
    y_bot = -0.58

    # Thin pixel line
    ax.plot(
        [x0, x2],
        [y_top, y_top],
        transform=ax.transAxes,
        color=thin_color,
        lw=0.8,
        solid_capstyle="round",
        clip_on=False,
        zorder=1,
    )

    ax.text(
        xt,
        y_top,
        thin_label,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.5,
        color="#333333",
        zorder=3,
    )

    # Thick source chunk line (blue)
    ax.plot(
        [x0, x1],
        [y_bot, y_bot],
        transform=ax.transAxes,
        color=thick_color,
        lw=1.8,
        solid_capstyle="round",
        clip_on=False,
        zorder=1,
    )

    # Thick destination chunk line (orange)
    ax.plot(
        [x1 + 0.01, x2 + 0.01],
        [y_bot, y_bot],
        transform=ax.transAxes,
        color=thick_color_2,
        lw=1.8,
        solid_capstyle="round",
        clip_on=False,
        zorder=1,
    )

    ax.text(
        xt,
        y_bot,
        thick_label,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.5,
        color="#333333",
        zorder=3,
    )

def build_demo_destination_grid(
    source_grid: GeoGrid,
    dst_crs: rio.crs.CRS,
    dst_shape: tuple[int, int],
) -> GeoGrid:
    """
    Build a destination GeoGrid from projected source bounds.
    """
    dst_bounds = source_grid.bounds_projected(dst_crs)
    dst_transform = rio.transform.from_bounds(
        west=dst_bounds.left,
        south=dst_bounds.bottom,
        east=dst_bounds.right,
        north=dst_bounds.top,
        width=dst_shape[1],
        height=dst_shape[0],
    )
    return GeoGrid(transform=dst_transform, shape=dst_shape, crs=dst_crs)


def plot_reprojection_chunk_diagram(
    source_chunked_grid: ChunkedGeoGrid,
    destination_chunked_grid: ChunkedGeoGrid,
    highlight_destination_chunk: tuple[int, int] = (1, 1),
    source_color: str = "#6BAED6",
    intersect_color: str = "#4C78A8",
    destination_color: str = "#F58518",
    figsize: tuple[float, float] = (10.0, 4.8),
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot a 2-panel diagram illustrating chunked reprojection.

    Left: Source raster with source chunks in source CRS.
    Right: Same source chunks projected to destination CRS, with destination chunk grid overlaid.
    """
    # -------------------------------------------------------------------------
    # Styling
    # -------------------------------------------------------------------------
    source_fill_alpha = 0.07
    source_chunk_alpha = 0.22
    intersect_alpha = 0.58

    grid_lw = 1.3
    highlight_lw = 2.6
    arrow_lw = 1.6
    pixel_grid_lw = 0.55

    neutral_color = "#333333"
    pixel_grid_color = "0.75"

    # -------------------------------------------------------------------------
    # Geometries
    # -------------------------------------------------------------------------
    src_full = source_chunked_grid.grid.footprint
    src_blocks = source_chunked_grid.get_block_footprints()

    dst_full = destination_chunked_grid.grid.footprint
    dst_blocks = destination_chunked_grid.get_block_footprints()

    src_full_in_dst = source_chunked_grid.grid.footprint_projected(destination_chunked_grid.grid.crs)
    src_blocks_in_dst = source_chunked_grid.get_block_footprints(crs=destination_chunked_grid.grid.crs)

    # -------------------------------------------------------------------------
    # Create figure
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(8.0, 3.5),
        gridspec_kw={"width_ratios": [0.75, 1]},
    )
    ax0, ax1 = axes

    # -------------------------------------------------------------------------
    # Left panel: source raster
    # -------------------------------------------------------------------------
    _add_rectilinear_grid(
        ax0,
        source_chunked_grid.grid,
        color=pixel_grid_color,
        linewidth=pixel_grid_lw,
        alpha=0.8,
        zorder=0,
    )

    _add_gdf(
        ax0,
        src_full,
        facecolor=source_color,
        edgecolor="none",
        linewidth=0,
        alpha=source_fill_alpha,
        zorder=1,
    )

    _add_gdf(
        ax0,
        src_blocks,
        facecolor=source_color,
        edgecolor="none",
        linewidth=0,
        alpha=source_chunk_alpha,
        zorder=2,
    )

    _add_gdf_boundary(
        ax0,
        src_blocks,
        color=source_color,
        linewidth=grid_lw,
        alpha=1,
        zorder=3,
    )

    src_blocks_list = source_chunked_grid.get_blocks_as_geogrids()
    ny, nx = source_chunked_grid.num_chunks

    bottom_left_chunk = src_blocks_list[(ny - 1) * nx]
    _add_chunk_size_arrows(
        ax0,
        bottom_left_chunk,
        label_x=f"{bottom_left_chunk.width} px",
        label_y=f"{bottom_left_chunk.height} px",
        color=neutral_color,
        fontsize=8.5,
    )

    ax0.text(
        0.5,
        0.95,
        "Source raster",
        transform=ax0.transAxes,
        ha="center",
        va="bottom",
        fontsize=12.5,
        fontweight="bold",
        color=source_color,
    )

    _set_ax_extent_from_gdfs(ax0, [src_full, src_blocks])

    _add_line_legend(
        ax1,
        thin_label="Pixel grid",
        thick_label="Chunk grid",
        thin_color=pixel_grid_color,
        thick_color=source_color,
        thick_color_2=destination_color,
    )

    # -------------------------------------------------------------------------
    # Right panel: projected source + destination raster
    # -------------------------------------------------------------------------
    _add_rectilinear_grid(
        ax1,
        destination_chunked_grid.grid,
        color=pixel_grid_color,
        linewidth=pixel_grid_lw,
        alpha=1,
        zorder=0,
    )

    _add_gdf(
        ax1,
        src_full_in_dst,
        facecolor=source_color,
        edgecolor="none",
        linewidth=0,
        alpha=source_fill_alpha,
        zorder=1,
    )

    highlight_index = destination_chunked_grid.flat_block_index(highlight_destination_chunk)
    dst_blocks_one = dst_blocks.iloc[[highlight_index]]
    highlight_geom = dst_blocks_one.geometry.iloc[0]

    intersection_area = src_blocks_in_dst.geometry.intersection(highlight_geom).area
    intersect_mask = intersection_area > 0

    src_intersect = src_blocks_in_dst[intersect_mask]
    src_non_intersect = src_blocks_in_dst[~intersect_mask]

    _add_gdf(
        ax1,
        src_non_intersect,
        facecolor=source_color,
        edgecolor="none",
        linewidth=0,
        alpha=source_chunk_alpha,
        zorder=2,
    )

    _add_gdf(
        ax1,
        src_intersect,
        facecolor=intersect_color,
        edgecolor="none",
        linewidth=0,
        alpha=intersect_alpha,
        zorder=3,
    )

    _add_gdf_boundary(
        ax1,
        src_blocks_in_dst,
        color=source_color,
        linewidth=grid_lw,
        alpha=0.9,
        zorder=4,
    )

    _add_gdf_boundary(
        ax1,
        dst_blocks,
        color=destination_color,
        linewidth=grid_lw,
        alpha=1.0,
        zorder=5,
    )

    _add_gdf(
        ax1,
        dst_blocks_one,
        facecolor="none",
        edgecolor=destination_color,
        linewidth=highlight_lw,
        alpha=1.0,
        zorder=6,
    )

    dst_blocks_list = destination_chunked_grid.get_blocks_as_geogrids()
    ny_d, nx_d = destination_chunked_grid.num_chunks

    bottom_left_dst_chunk = dst_blocks_list[(ny_d - 1) * nx_d]
    _add_chunk_size_arrows(
        ax1,
        bottom_left_dst_chunk,
        label_x=f"{bottom_left_dst_chunk.width} px",
        label_y=f"{bottom_left_dst_chunk.height} px",
        color=neutral_color,
        fontsize=8.5,
    )

    _set_ax_extent_from_gdfs(ax1, [src_full_in_dst, src_blocks_in_dst, dst_full, dst_blocks])

    # -------------------------------------------------------------------------
    # Annotation for highlighted destination chunk
    # -------------------------------------------------------------------------
    n_intersections = len(src_intersect)
    target = dst_blocks_one.geometry.iloc[0].centroid
    tx, ty = target.x, target.y

    from matplotlib.offsetbox import AnnotationBbox, HPacker, VPacker, TextArea

    # Build colored text pieces
    line1 = HPacker(
        children=[
            TextArea("Destination chunk", textprops=dict(color=destination_color, fontsize=9)),
        ],
        align="center",
        pad=0,
        sep=2,
    )

    line2 = HPacker(
        children=[
            TextArea("requires", textprops=dict(color="black", fontsize=9)),
        ],
        align="center",
        pad=0,
        sep=2,
    )
    line3 = HPacker(
        children=[
            TextArea(f"{n_intersections}", textprops=dict(color=intersect_color, fontsize=9, fontweight="bold")),
            TextArea(" source chunks", textprops=dict(color=intersect_color, fontsize=9)),
        ],
        align="center",
        pad=0,
        sep=1,
    )

    label_box = VPacker(
        children=[line1, line2, line3],
        align="center",
        pad=0,
        sep=2,
    )

    ann = AnnotationBbox(
        label_box,
        (tx, ty),
        xycoords="data",
        xybox=(0.2, -0.5),
        boxcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="-|>",
            color="black",
            lw=arrow_lw,
            shrinkA=0,
            shrinkB=1,
            mutation_scale=5,
            connectionstyle="arc3,rad=0",
        ),
        bboxprops=dict(
            boxstyle="round,pad=0.3,rounding_size=0.15",
            fc="white",
            ec="0.85",
            lw=0.8,
            alpha=0.96,
        ),
    )

    ax1.add_artist(ann)

    # Ensure annotation is above the chunk polygons
    ann.set_zorder(20)

    if ann.arrow_patch is not None:
        ann.arrow_patch.set_zorder(21)
        ann.arrow_patch.set_clip_on(False)

    if ann.patch is not None:
        ann.patch.set_zorder(20)

    # -------------------------------------------------------------------------
    # Panel labels
    # -------------------------------------------------------------------------
    ax1.text(
        0.5,
        1.01,
        "Destination raster",
        transform=ax1.transAxes,
        ha="center",
        va="bottom",
        fontsize=12.5,
        fontweight="bold",
        color=destination_color,
    )

    ax1.text(
        0.5,
        -0.02,
        "Projected source raster",
        transform=ax1.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color=source_color,
        style="italic",
        alpha=0.95,
    )

    # -------------------------------------------------------------------------
    # Grouped title + underline + reprojection arrow
    # -------------------------------------------------------------------------
    src_crs = source_chunked_grid.grid.crs.to_epsg()
    dst_crs = destination_chunked_grid.grid.crs.to_epsg()

    # Finalize layout before placing figure-level decorations
    fig.canvas.draw()

    pos_left = ax0.get_position()
    pos_right = ax1.get_position()

    x_center = 0.5 * (pos_left.x0 + pos_right.x1)
    y_text = pos_left.y1 + 0.085
    y_line = pos_left.y1 + 0.077

    # Grey grouped title
    fig.text(
        x_center,
        y_text,
        "Chunked reprojection",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="semibold",
        color="0.35",
    )

    # Underline spanning both panels
    line = LineCollection(
        [np.array([[pos_left.x0, y_line], [pos_right.x1, y_line]])],
        colors="0.6",
        linewidths=1.0,
        alpha=0.9,
        transform=fig.transFigure,
        zorder=100,
    )
    fig.add_artist(line)

    # Curved arrow and label between panels
    label = f"EPSG:{src_crs} → EPSG:{dst_crs}\nResolution × 2"

    arrow = FancyArrowPatch(
        (0.455, y_line - 0.16),
        (0.545, y_line - 0.16),
        transform=fig.transFigure,
        arrowstyle="-|>",
        connectionstyle="arc3,rad=-0.35",
        linewidth=1.7,
        color=neutral_color,
        mutation_scale=20,
    )
    fig.add_artist(arrow)

    fig.text(
        0.50,
        y_line - 0.11,
        label,
        ha="center",
        va="bottom",
        fontsize=10,
        color=neutral_color,
    )

    bbox_info = dict(
        boxstyle="round,pad=0.35,rounding_size=0.2",
        fc="white",
        ec="0.85",
        lw=0.8,
        alpha=0.95,
    )

    ax1.text(
        0.7,
        1.65,
        "Chunk size adapts "
        "\nto resolution change\n"
        "4×4 px → 2×2 px",
        transform=ax1.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="0.35",
        bbox=bbox_info,
        zorder=10,
    )
    # -------------------------------------------------------------------------
    # Final layout polish
    # -------------------------------------------------------------------------
    fig.subplots_adjust(
        left=0,
        right=1,
        top=0.88,
        bottom=0.03,
        wspace=0.03,
    )

    return fig, axes

# -----------------------------------------------------------------------------
# Main figure
# -----------------------------------------------------------------------------

# Source grid in a projected CRS, with a fairly large extent to accentuate
# visible deformation after reprojection to geographic coordinates.
src_grid = GeoGrid(
    transform=rio.transform.from_origin(
        west=350_000,
        north=8_300_000,
        xsize=8_000,
        ysize=8_000,
    ),
    shape=(20, 20),
    crs=rio.crs.CRS.from_epsg(32633),  # UTM 33N
)

# Regular chunking for a cleaner schematic
# Use chunk sizes divisible by 2 so the destination chunking can be scaled
# consistently with the resolution change.
src_chunks = (
    (4, 4, 4, 4, 4),
    (4, 4, 4, 4, 4),
)
src_chunked = ChunkedGeoGrid(grid=src_grid, chunks=src_chunks)

# Destination grid: same extent, but 2x coarser resolution in both dimensions
# Since the source is 20x20 pixels, the destination becomes 10x10 pixels.
dst_grid = build_demo_destination_grid(
    source_grid=src_grid,
    dst_crs=rio.crs.CRS.from_epsg(4326),
    dst_shape=(10, 10),
)

# Scale chunk sizes with the resolution change so chunk footprints remain
# approximately comparable between source and destination.
# Source chunk size of 4x4 so we'll have a destination chunk size of 2x2 px
dst_chunks = (
    (2, 2, 2, 2, 2),
    (2, 2, 2, 2, 2),
)
dst_chunked = ChunkedGeoGrid(grid=dst_grid, chunks=dst_chunks)

fig, _ = plot_reprojection_chunk_diagram(
    source_chunked_grid=src_chunked,
    destination_chunked_grid=dst_chunked,
    highlight_destination_chunk=(1, 1),
)
plt.show()
