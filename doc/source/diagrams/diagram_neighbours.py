"""Illustrate point count and radius limits alongside grid window sizes and shapes."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

from geoutils.operators import GridNeighbours, PointNeighbours

NEUTRAL = "#253246"
SELECTED = "#4C78A8"
FADED = "#C8D0D8"
TARGET = "#D97732"


def draw_point_neighbours(ax: plt.Axes, points: np.ndarray, neighborhood: PointNeighbours) -> None:
    """Draw the nearest observations, limited by radius and max_points."""

    # Apply the radius limit before the point count limit
    distances = np.linalg.norm(points, axis=1)
    candidates = np.argsort(distances)
    assert neighborhood.radius is not None and neighborhood.max_points is not None
    candidates = candidates[distances[candidates] <= neighborhood.radius][: neighborhood.max_points]
    ax.scatter(*points.T, s=52, color=FADED, edgecolor="white", linewidth=0.8, zorder=2)
    ax.scatter(*points[candidates].T, s=56, color=SELECTED, edgecolor="white", linewidth=0.8, zorder=3)

    radius = neighborhood.radius
    ax.add_patch(Circle((0, 0), radius, fill=False, edgecolor=TARGET, linewidth=1.5))
    ax.plot([0, 0], [0, -radius], color=TARGET, linewidth=1)
    ax.text(0.13, -radius * 0.62, "radius", fontsize=11, color=TARGET)
    ax.scatter(0, 0, marker="+", s=100, color=TARGET, linewidth=1.8, zorder=4)
    ax.set_aspect("equal")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_axis_off()


def draw_grid_neighbours(ax: plt.Axes, size: int) -> None:
    """Draw a circular window and the surrounding grid cells outside it."""

    # Include a ring of excluded cells around the window
    neighborhood = GridNeighbours(size=size, shape="circular")
    middle = size // 2
    selected = set(neighborhood.offsets)
    for row in range(-1, size + 1):
        for col in range(-1, size + 1):
            inside = (row - middle, col - middle) in selected
            if inside:
                ax.add_patch(Rectangle((col, row), 1, 1, facecolor=SELECTED, alpha=0.16, edgecolor="none"))
            ax.scatter(col + 0.5, row + 0.5, s=22, color=SELECTED if inside else FADED, zorder=3)
    for edge in range(-1, size + 2):
        ax.plot([edge, edge], [-1, size + 1], color=FADED, linewidth=0.65, zorder=1)
        ax.plot([-1, size + 1], [edge, edge], color=FADED, linewidth=0.65, zorder=1)

    # Match the circle used in the coverage diagram
    center = size / 2
    ax.add_patch(Circle((center, center), size / 2, fill=False, edgecolor=TARGET, linewidth=1.5, zorder=4))
    ax.scatter(center, center, marker="+", s=100, color=TARGET, linewidth=1.8, zorder=5)
    ax.set_aspect("equal")
    ax.set_xlim(-1.15, size + 1.15)
    ax.set_ylim(-1.15, size + 1.15)
    ax.set_axis_off()


def make_neighbours_diagram() -> tuple[plt.Figure, np.ndarray]:
    """Compare one bounded search on irregular points with one circular window on a grid."""

    # Use a few points so the excluded observations inside the radius remain clear
    points = np.array(
        [
            [-0.35, 0.4],
            [0.6, 0.1],
            [-0.2, -0.8],
            [0.9, -0.55],
            [-1.2, -0.25],
            [0.3, 1.4],
            [-1.85, 1.1],
            [1.9, 1.45],
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    draw_point_neighbours(axes[0], points, PointNeighbours(max_points=4, radius=1.5))
    draw_grid_neighbours(axes[1], size=5)

    titles = ("Point", "Grid (raster)")
    descriptions = ("max_points=4, radius=1.5", 'window=5, window_shape="circular"')
    for ax, title, description in zip(axes, titles, descriptions, strict=True):
        ax.set_title(title, fontsize=13, color=NEUTRAL, pad=29)
        ax.text(
            0.5, 1.05, description, transform=ax.transAxes, ha="center", fontsize=11, color=NEUTRAL, family="monospace"
        )
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=SELECTED, markersize=6, label="Selected"),
        Line2D([], [], marker="o", linestyle="none", color=FADED, markersize=6, label="Excluded"),
        Line2D([], [], marker="+", linestyle="none", color=TARGET, markersize=8, label="Target"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=11, handletextpad=0.5)
    fig.subplots_adjust(left=0.04, right=0.96, top=0.77, bottom=0.17, wspace=0.3)
    return fig, axes


fig, _ = make_neighbours_diagram()
plt.show()
