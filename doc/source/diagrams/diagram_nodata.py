"""Compare a mean that ignores a missing observation with one that propagates it."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from affine import Affine
from matplotlib.patches import Rectangle

import geoutils as gu
from geoutils.operators import GridNeighbours, Interpolator, LocalData

NEUTRAL = "#253246"
SELECTED = "#4C78A8"
FADED = "#C8D0D8"
TARGET = "#D97732"
MISSING = "#B84C55"


class _WindowMean(Interpolator):
    """Predict each shifted output from the mean of valid values in its source window."""

    def predict(self, data: LocalData) -> float:
        """Average the observations selected by the neighborhood and nodata rule."""
        return float(np.mean(data.values))


def draw_nodata_panel(
    ax: plt.Axes,
    values: np.ndarray,
    positions: np.ndarray,
    results: np.ndarray,
    rule: str | int,
    description: str,
) -> None:
    """Draw source cells, shifted outputs, and the middle output's five-cell neighborhood."""

    # Shade the source mask, with expansion for the distance rule
    if rule in ("nearest", 1):
        spread_distance = 1 if rule == 1 else 0
        left, right = 2 - spread_distance, 3 + spread_distance
        ax.add_patch(Rectangle((left, -0.6), right - left, 0.85, facecolor=MISSING, alpha=0.08))
        for boundary in (left, right):
            ax.plot([boundary, boundary], [-0.6, 0.25], color=MISSING, linewidth=0.8, linestyle="dashed")

    for column, value in enumerate(values):
        color = SELECTED if np.isfinite(value) else MISSING
        ax.add_patch(Rectangle((column, 0.3), 1, 0.8, facecolor=color, alpha=0.16, edgecolor="none"))
        ax.add_patch(Rectangle((column, 0.3), 1, 0.8, fill=False, edgecolor=FADED, linewidth=0.8))
        label = f"{value:.0f}" if np.isfinite(value) else "—"
        ax.text(column + 0.5, 0.7, label, ha="center", va="center", fontsize=12, color=color)

    # Show the middle output's window separately from mask expansion
    ax.add_patch(Rectangle((-0.04, 0.25), 5.08, 0.9, fill=False, edgecolor=SELECTED, linestyle="dashed"))
    for left, right in ((0.5, 2.5), (2.5, 4.5)):
        ax.annotate(
            "",
            xy=(left, 1.45),
            xytext=(right, 1.45),
            arrowprops={"arrowstyle": "|-|", "color": SELECTED, "linewidth": 0.9},
        )
        ax.text((left + right) / 2, 1.58, "2 pixels", ha="center", fontsize=9, color=SELECTED)
    ax.annotate(
        "",
        xy=(positions[2], -0.6),
        xytext=(2.5, 0.17),
        arrowprops={"arrowstyle": "->", "color": NEUTRAL, "linewidth": 1.1},
    )

    # Outputs are shifted a quarter pixel to the right
    for position, result in zip(positions, results, strict=True):
        color = TARGET if np.isfinite(result) else MISSING
        ax.add_patch(Rectangle((position - 0.36, -1.45), 0.72, 0.8, facecolor=color, alpha=0.16, edgecolor="none"))
        label = f"{result:.1f}" if np.isfinite(result) else "NaN"
        ax.text(position, -1.05, label, ha="center", va="center", fontsize=10, color=color)

    if rule == 1:
        for left, right in ((1, 2), (3, 4)):
            ax.annotate(
                "",
                xy=(left, -0.15),
                xytext=(right, -0.15),
                arrowprops={"arrowstyle": "<->", "color": MISSING, "linewidth": 1},
            )
            ax.text((left + right) / 2, -0.45, "1 pixel", ha="center", fontsize=8, color=MISSING)

    ax.text(-0.12, 0.7, "Input", ha="right", va="center", fontsize=9, color=NEUTRAL)
    ax.text(-0.12, -1.05, "Output", ha="right", va="center", fontsize=9, color=NEUTRAL)
    title = "Distance: 1 pixel" if rule == 1 else f'"{rule}"'
    ax.set_title(title, fontsize=13, color=NEUTRAL, pad=15)
    ax.text(2.5, -1.95, description, ha="center", fontsize=9, color=NEUTRAL)
    ax.set_xlim(-0.9, 5.3)
    ax.set_ylim(-2.15, 1.85)
    ax.set_axis_off()


def make_nodata_diagram() -> tuple[plt.Figure, np.ndarray]:
    """Show the same missing observation being omitted or propagated through a mean."""

    # The valid observations have a mean of 4.5
    values = np.array([3.0, 5.0, np.nan, 7.0, 3.0])
    raster = gu.Raster.from_array(values[None, :], Affine(1, 0, 0, 0, -1, 1), 32631, nodata=-9999)
    positions = np.arange(len(values), dtype=float) + 0.75
    locations = positions, np.full_like(positions, 0.5)
    neighborhood = GridNeighbours(offsets=tuple((0, offset) for offset in range(-2, 3)))
    operator = _WindowMean(neighborhood=neighborhood)

    # Change only the nodata rule; edge windows use the available cells
    rules: tuple[str | int, ...] = ("ignore", "propagate", "nearest", 1)
    descriptions = (
        "Average the valid values in each window",
        "Every window includes the missing value",
        "Only the middle output has a missing nearest source",
        "Expand the source mask by 1 pixel on each side",
    )
    predictions = [
        raster.resample_at_points(locations, operator, nodata_handling=rule, as_array=True) for rule in rules
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4))

    for ax, rule, result, description in zip(axes.flat, rules, predictions, descriptions, strict=True):
        draw_nodata_panel(ax, values, positions, result, rule, description)

    fig.text(
        0.5,
        0.025,
        "Dashed window: middle output's neighborhood; edge windows use available cells.\n"
        "Outputs are shifted by ¼ source pixel.",
        ha="center",
        fontsize=9,
        color=NEUTRAL,
    )
    fig.subplots_adjust(left=0.03, right=0.98, top=0.90, bottom=0.09, wspace=0.22, hspace=0.45)
    return fig, axes


fig, _ = make_nodata_diagram()
plt.show()
