"""Compare a curved prediction and a mean using five selected observations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline

NEUTRAL = "#253246"
SELECTED = "#4C78A8"
TARGET = "#D97732"


def make_operators_diagram() -> tuple[plt.Figure, np.ndarray]:
    """Compare cubic interpolation at a central target and a mean of five observations."""

    # Put the target between observations so interpolation has a visible effect
    positions = np.array([1.0, 2.0, 3.5, 4.0, 5.0])
    values = np.array([3.0, 5.0, 8.0, 7.0, 3.0])
    target = (positions[0] + positions[-1]) / 2
    spline = CubicSpline(positions, values, bc_type="natural")
    estimates = (float(spline(target)), float(np.mean(values)))
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.6), sharex=True, sharey=True)

    curve_positions = np.linspace(positions[0], positions[-1], 200)
    axes[0].plot(curve_positions, spline(curve_positions), color=SELECTED, linewidth=1.8, zorder=2)
    axes[1].hlines(estimates[1], positions[0], positions[-1], color=TARGET, linewidth=1.5, zorder=2)

    # Label observations and predictions directly
    titles = ("Interpolator: predict at a coordinate", "Reducer: summarize values")
    descriptions = ("Cubic spline", "Mean")
    for ax, title, description, estimate in zip(axes, titles, descriptions, estimates, strict=True):
        ax.scatter(
            positions,
            values,
            s=48,
            color=SELECTED,
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        for position, value in zip(positions, values, strict=True):
            ax.annotate(
                f"{value:.0f}",
                (position, value),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=11,
                color=NEUTRAL,
            )
        ax.vlines(target, 0, estimate, color=TARGET, linewidth=1, linestyles="dashed", alpha=0.55)
        ax.scatter(target, estimate, s=65, color=TARGET, edgecolor="white", linewidth=1, zorder=4)
        ax.annotate(
            f"{estimate:.1f}",
            (target, estimate),
            xytext=(10, -12),
            textcoords="offset points",
            fontsize=11,
            color=TARGET,
        )
        ax.set_title(title, fontsize=13, color=NEUTRAL, pad=28)
        ax.text(0.5, 1.045, description, transform=ax.transAxes, ha="center", fontsize=11, color=NEUTRAL)
        ax.set_xlim(positions[0] - 0.3, positions[-1] + 0.3)
        ax.set_ylim(0, 10.3)
        ax.set_axis_off()

    handles = [
        Line2D([], [], marker="o", linestyle="none", color=SELECTED, markersize=6, label="Observations"),
        Line2D([], [], marker="o", linestyle="none", color=TARGET, markersize=6, label="Result at target"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10, handletextpad=0.5)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.13, hspace=0.65)
    return fig, axes


fig, _ = make_operators_diagram()
plt.show()
