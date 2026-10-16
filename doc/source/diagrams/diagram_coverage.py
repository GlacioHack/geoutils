"""Illustrate center, all touched and fractional coverage on the same raster grid."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from shapely.geometry import Point, box

NEUTRAL = "#253246"
SELECTED = "#4C78A8"
FADED = "#C8D0D8"
TARGET = "#D97732"


def make_coverage_diagram() -> tuple[plt.Figure, np.ndarray]:
    """Compare three coverage rules for a five-cell circle, labeling every fractional weight."""

    # Reuse the circular grid window from the neighborhood diagram
    size = 5
    center = size / 2
    circle = Point(center, center).buffer(size / 2, quad_segs=256)
    modes = ("center", "all_touched", "fractional")
    titles = ("Center", "All touched", "Fractional")
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 3.0))

    # Use color intensity to show each cell's contribution
    for ax, mode, title in zip(axes, modes, titles, strict=True):
        for row in range(size):
            for col in range(size):
                cell = box(col, row, col + 1, row + 1)
                cell_center = Point(col + 0.5, row + 0.5)
                if mode == "center":
                    weight = float(circle.contains(cell_center))
                elif mode == "all_touched":
                    weight = float(circle.intersects(cell))
                else:
                    weight = circle.intersection(cell).area / cell.area
                if weight > 0:
                    ax.add_patch(Rectangle((col, row), 1, 1, facecolor=SELECTED, alpha=0.38 * weight, edgecolor="none"))
                if mode in ("center", "all_touched"):
                    ax.scatter(cell_center.x, cell_center.y, s=18, color=SELECTED if weight else FADED, zorder=3)

                if mode == "fractional":
                    # Shift edge labels away from the circle outline
                    label_x = col + 0.5
                    if row in (1, size - 2) and col == 0:
                        label_x += 0.12
                    elif row in (1, size - 2) and col == size - 1:
                        label_x -= 0.12
                    ax.text(
                        label_x,
                        row + 0.5,
                        f"{weight:.0%}",
                        ha="center",
                        va="center",
                        fontsize=10,
                        color=NEUTRAL,
                        zorder=5,
                    )

        # Use the same circle and grid in all panels
        for edge in range(size + 1):
            ax.plot([edge, edge], [0, size], color=FADED, linewidth=0.7, zorder=2)
            ax.plot([0, size], [edge, edge], color=FADED, linewidth=0.7, zorder=2)
        ax.add_patch(Circle((center, center), size / 2, fill=False, edgecolor=TARGET, linewidth=1.5, zorder=3))
        ax.set_aspect("equal")
        ax.set_xlim(-0.1, size + 0.1)
        ax.set_ylim(-0.1, size + 0.1)
        ax.set_axis_off()
        ax.set_title(title, fontsize=13, color=NEUTRAL, pad=16)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.8, bottom=0.05, wspace=0.23)
    return fig, axes


fig, _ = make_coverage_diagram()
plt.show()
