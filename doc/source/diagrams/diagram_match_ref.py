"""Compare the reference attributes used by raster reprojection and cropping."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from affine import Affine
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from rasterio.transform import from_bounds

import geoutils as gu

NEUTRAL = "#253246"
SOURCE = "#4C78A8"
FADED = "#EAF0F5"
REFERENCE = "#D97732"


def draw_raster(ax: plt.Axes, raster: gu.Raster, reference: bool = False) -> None:
    """
    Draw cell values in blue or an orange reference grid in native coordinates.

    Geographic grids have a schematic skew to make their different CRS visible.
    """

    # Raster rows run from top to bottom
    left, bottom, right, top = raster.bbox
    x_edges = np.linspace(left, right, raster.width + 1)
    y_edges = np.linspace(top, bottom, raster.height + 1)
    color = REFERENCE if reference else SOURCE
    colors = ("#FFF4EA", "#FFF4EA") if reference else (FADED, SOURCE)
    colormap = LinearSegmentedColormap.from_list("raster_cells", colors)

    # Apply the skew only when drawing, not when calculating the results
    drawing_transform = ax.transData
    if raster.crs.is_geographic:
        center_x, center_y = (left + right) / 2, (bottom + top) / 2
        drawing_transform = (
            Affine2D().translate(-center_x, -center_y).skew_deg(-20, 0).translate(center_x, center_y) + ax.transData
        )

    # Use the same color scale for the source and result
    ax.pcolormesh(
        x_edges,
        y_edges,
        raster.data,
        cmap=colormap,
        vmin=0,
        vmax=1,
        edgecolors=color,
        linewidth=0.6,
        transform=drawing_transform,
    )
    ax.add_patch(
        Rectangle(
            (left, bottom),
            right - left,
            top - bottom,
            fill=False,
            edgecolor=color,
            linewidth=1.2,
            transform=drawing_transform,
        )
    )


def make_match_ref_diagram() -> tuple[plt.Figure, np.ndarray]:
    """
    Show reprojection matching a reference grid and cropping selecting source cells within its bounding box.

    We calculate both results with the public Raster methods, using the same geographic reference. Its bounding box aligns
    with the crop source's coarser cells so no rounding is needed. draw_raster() displays each grid in its own CRS;
    the projected panels share an extent so their pixel sizes can be compared directly.
    """

    # Create a coarse projected raster and a finer reference covering its central nine cells
    rows, columns = np.indices((5, 5))
    values = np.exp(-((rows - 2) ** 2 + (columns - 2) ** 2) / 4).astype(np.float32)
    projected_reference = gu.Raster.from_array(
        np.zeros((6, 6), dtype=np.float32), Affine(100, 0, 650200, 0, -100, 4999800), 32631
    )

    # Cover the same area with a geographic reference
    geographic_bounds = projected_reference.get_bbox_projected(out_crs=4326)
    geographic_transform = from_bounds(*geographic_bounds, width=6, height=6)
    reference = gu.Raster.from_array(np.zeros((6, 6), dtype=np.float32), geographic_transform, 4326)

    # Align the reference bounding box with the central 3 × 3 source cells
    projected_bounds = reference.get_bbox_projected(out_crs=32631)
    left, bottom, right, top = projected_bounds
    cell_width, cell_height = (right - left) / 3, (top - bottom) / 3
    source_bounds = (left - cell_width, bottom - cell_height, right + cell_width, top + cell_height)
    source_transform = from_bounds(*source_bounds, width=5, height=5)
    source = gu.Raster.from_array(values, source_transform, 32631)

    reprojected = source.reproject(reference)
    cropped = source.crop(reference)
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 5.6))

    rasters = ((source, reference, reprojected), (source, reference, cropped))
    descriptions = (
        ("Projected CRS · metres", "Geographic CRS · degrees", "Matches CRS + grid"),
        ("Projected CRS · metres", "Geographic CRS · degrees", "Matches bounding box\nin source CRS"),
    )
    geographic_source_bounds = source.get_bbox_projected(out_crs=reference.crs)
    for row_axes, row_rasters in zip(axes, rasters, strict=True):
        for column, (ax, raster) in enumerate(zip(row_axes, row_rasters, strict=True)):
            draw_raster(ax, raster, reference=column == 1)

            # Share extents within each CRS so cell sizes can be compared
            bounds = geographic_source_bounds if raster.crs.is_geographic else source.bbox
            left, bottom, right, top = bounds
            x_padding = 0.18 if raster.crs.is_geographic else 0.04
            ax.set(xlim=(left - (right - left) * x_padding, right + (right - left) * x_padding))
            ax.set(ylim=(bottom - (top - bottom) * 0.04, top + (top - bottom) * 0.04), aspect="equal")
            ax.set_axis_off()

    # Mark the reference bounding box on the crop source and both results
    for ax, raster in ((axes[1, 0], source), (axes[0, 2], reprojected), (axes[1, 2], cropped)):
        left, bottom, right, top = reference.get_bbox_projected(out_crs=raster.crs)
        ax.add_patch(
            Rectangle(
                (left, bottom),
                right - left,
                top - bottom,
                fill=False,
                edgecolor=REFERENCE,
                linestyle=(0, (4, 3)),
                linewidth=1.6,
                transform=ax.collections[0].get_transform(),
            )
        )

    fig.subplots_adjust(left=0.085, right=0.975, top=0.88, bottom=0.14, hspace=0.45, wspace=0.85)

    # Align captions even when the grids have different aspect ratios
    for row_axes, row_descriptions in zip(axes, descriptions, strict=True):
        caption_y = row_axes[0].get_position(original=True).y0 - 0.01
        for ax, description in zip(row_axes, row_descriptions, strict=True):
            column_box = ax.get_position()
            fig.text(
                column_box.x0 + column_box.width / 2,
                caption_y,
                description,
                ha="center",
                va="top",
                fontsize=10,
                color=NEUTRAL,
            )

    for ax, title in zip(axes[0], ("Source", "Reference", "Output"), strict=True):
        column_box = ax.get_position()
        fig.text(
            column_box.x0 + column_box.width / 2,
            0.955,
            title,
            ha="center",
            va="center",
            fontsize=13,
            color=REFERENCE if title == "Reference" else NEUTRAL,
        )

    # Label each operation beside its row
    for row_axes, label, operation in zip(
        axes, ("Reproject", "Crop"), ("source.reproject(ref)", "source.crop(ref)"), strict=True
    ):
        source_box, reference_box, result_box = (ax.get_position() for ax in row_axes)
        middle_y = reference_box.y0 + reference_box.height / 2
        fig.text(
            0.028,
            middle_y,
            label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=NEUTRAL,
        )

        fig.text(
            (source_box.x1 + reference_box.x0) / 2, middle_y, "+", ha="center", va="center", fontsize=17, color=NEUTRAL
        )
        row_axes[1].annotate(
            "",
            xy=(result_box.x0 - 0.014, middle_y),
            xytext=(reference_box.x1 + 0.014, middle_y),
            xycoords=fig.transFigure,
            arrowprops=dict(arrowstyle="->", color=NEUTRAL, linewidth=1.2),
        )
        fig.text(
            (reference_box.x1 + result_box.x0) / 2,
            middle_y + 0.045,
            operation,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=NEUTRAL,
            family="monospace",
        )

    separator_y = (axes[0, 1].get_position().y0 + axes[1, 1].get_position().y1) / 2
    fig.add_artist(
        Line2D(
            (0.015, 0.975),
            (separator_y, separator_y),
            transform=fig.transFigure,
            color=NEUTRAL,
            alpha=0.25,
            linewidth=0.6,
        )
    )

    fig.legend(
        handles=[Line2D([], [], color=REFERENCE, linestyle=(0, (4, 3)), linewidth=1.6)],
        labels=["Reference bounding box (in source CRS for cropping)"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        frameon=False,
        fontsize=10,
        labelcolor=NEUTRAL,
    )

    return fig, axes


fig, _ = make_match_ref_diagram()
plt.show()
