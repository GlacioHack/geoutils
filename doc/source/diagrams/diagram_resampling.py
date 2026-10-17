"""Compare all built-in interpolators and reducers on a one-dimensional raster profile."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from affine import Affine
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea

import geoutils as gu
from geoutils.operators import GridNeighbours, Interpolator, Reducer
from geoutils.operators.interpolator import (
    Cubic,
    InverseDistance,
    Kriging,
    Linear,
    Nearest,
    RasterConvolution,
    ScipyInterpolator,
)
from geoutils.operators.reducer import (
    AverageDistance,
    AveragePairwiseDistance,
    Count,
    Maximum,
    Mean,
    Median,
    Minimum,
    Mode,
    Quantile,
    Range,
    RootMeanSquare,
    StandardDeviation,
    Sum,
)
from geoutils.stats.variography import VariogramModel

NEUTRAL = "#253246"
SELECTED = "#4C78A8"
FADED = "#C8D0D8"
TARGET = "#D97732"
POSITIONS = np.arange(8, dtype=float) + 0.5
VALUES = np.array([2.0, 5.0, 3.0, 3.0, 7.0, 1.0, 4.0, 2.0])
TARGET_POSITION = 3.8
POINT_NUMBA_METHODS = {"nearest", "idw", "mean", "minimum", "maximum", "range", "std", "count", "average_distance"}
GRID_NUMBA_METHODS = {"mean", "sum", "median", "minimum", "maximum", "range", "rms", "count"}
GDAL_METHODS = {
    "nearest",
    "linear",
    "cubic",
    "cubic_convolution",
    "cubic_spline",
    "lanczos",
    "mean",
    "sum",
    "median",
    "q1",
    "q3",
    "mode",
    "minimum",
    "maximum",
    "rms",
}


def draw_profile(ax: plt.Axes, selected: np.ndarray) -> None:
    """Draw the same observations and target, highlighting the selected source cells."""

    colors = np.where(selected, SELECTED, FADED)
    ax.scatter(POSITIONS, VALUES, s=30, color=colors, edgecolor="white", linewidth=0.7, zorder=3)
    ax.axvline(TARGET_POSITION, color=TARGET, linewidth=0.9, linestyle="dashed", alpha=0.5)
    ax.set_xlim(0, 8)
    ax.set_axis_off()


def draw_method_title(ax: plt.Axes, method: str, supports_points: bool) -> None:
    """Label a panel with its method and numerical backends for GRID or POINT observations."""

    # Leave room between long method names and the input badges
    ax.set_title(f'"{method}"', fontsize=10, fontfamily="monospace", color=NEUTRAL, pad=44)
    tags = ("RASTER", "POINT")
    for index, tag in enumerate(tags):
        position = 0.5 + (index - (len(tags) - 1) / 2) * 0.60
        ax.text(
            position,
            1.55,
            tag,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7,
            color=NEUTRAL,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "#E5EEF6" if tag == "RASTER" else "#EEE7F4",
                "edgecolor": "none",
            },
        )

        if tag == "POINT" and not supports_points:
            backends = ["—"]
        elif method == "kriging":
            backends = ['"gstools"', '"gpytorch"']
        else:
            backends = ['"scipy"']
            numba_methods = GRID_NUMBA_METHODS if tag == "RASTER" else POINT_NUMBA_METHODS
            if method in numba_methods:
                backends.append('"numba"')
            if tag == "RASTER" and method in GDAL_METHODS:
                backends.append('"gdal"*')

        ax.text(
            position,
            1.47,
            "\n".join(backends),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=6.5,
            fontfamily="monospace",
            color=NEUTRAL,
        )


def _add_input_legend(fig: plt.Figure) -> None:
    """Explain the input badges and GDAL restriction below a method comparison."""

    labels = []
    for name, color in (("RASTER", "#E5EEF6"), ("POINT", "#EEE7F4")):
        labels.append(
            TextArea(
                name,
                textprops={
                    "fontsize": 8,
                    "color": NEUTRAL,
                    "bbox": {"boxstyle": "round,pad=0.3", "facecolor": color, "edgecolor": "none"},
                },
            )
        )
    labels.append(TextArea("Input data being resampled", textprops={"fontsize": 8, "color": NEUTRAL}))

    legend = HPacker(children=labels, align="center", pad=0, sep=10)
    fig.add_artist(
        AnnotationBbox(legend, (0.5, 0.042), xycoords="figure fraction", box_alignment=(0.5, 0), frameon=False)
    )
    fig.text(0.5, 0.012, '"gdal"*: reproject() only', ha="center", fontsize=7, color=NEUTRAL)


def make_interpolator_diagram() -> plt.Figure:
    """
    Predict the same profile with every built-in interpolation method and mark its value at x = 3.8.

    We repeat the profile along raster rows so regular 2D methods produce a 1D cross-section. IDW and kriging
    use only the displayed row. draw_profile() adds the observations and the common target to each panel.
    """

    # Align SciPy's corner coordinates with local operators' cell centers
    values = np.tile(VALUES, (9, 1))
    raster = gu.Raster.from_array(values, Affine(1, 0, 0, 0, -1, 9), 32631, nodata=-9999)
    regular_raster = gu.Raster.from_array(values, Affine(1, 0, 0.5, 0, -1, 9.5), 32631, nodata=-9999)
    row = GridNeighbours(offsets=tuple((0, offset) for offset in range(-8, 9)))
    model = VariogramModel("gaussian", effective_range=3, partial_sill=1)
    operators: list[tuple[str, Interpolator, bool]] = [
        ("nearest", Nearest(), True),
        ("linear", Linear(), True),
        ("cubic", Cubic(), True),
        ("slinear", ScipyInterpolator("slinear"), False),
        ("quintic", ScipyInterpolator("quintic"), False),
        ("pchip", ScipyInterpolator("pchip"), False),
        ("splinef2d", ScipyInterpolator("splinef2d"), False),
        ("cubic_convolution", RasterConvolution("cubic"), False),
        ("cubic_spline", RasterConvolution("cubic_spline"), False),
        ("lanczos", RasterConvolution("lanczos"), False),
        ("idw", InverseDistance(neighborhood=row), True),
        ("kriging", Kriging(model, neighborhood=row), True),
    ]

    # Append the target to each profile query to calculate both in one call
    positions = np.linspace(POSITIONS[0] + 0.01, POSITIONS[-1] - 0.01, 140)
    predictions = []
    for method, operator, supports_points in operators:
        # Sampling nearest at cell centers puts its steps on cell boundaries
        profile_positions = POSITIONS if isinstance(operator, Nearest) else positions
        queries = np.append(profile_positions, TARGET_POSITION)
        locations = queries, np.full_like(queries, 4.5)
        source = regular_raster if isinstance(operator, (ScipyInterpolator, Nearest)) else raster
        result = source.resample_at_points(locations, operator, nodata_handling="ignore", as_array=True)
        predictions.append((method, profile_positions, result, supports_points))
    limits = np.concatenate([result for _, _, result, _ in predictions])
    lower, upper = min(0, np.nanmin(limits)) - 0.6, max(VALUES.max(), np.nanmax(limits)) + 1.5
    fig, axes = plt.subplots(3, 4, figsize=(8.6, 6.3), sharex=True, sharey=True)

    for ax, (method, profile_positions, result, supports_points) in zip(axes.flat, predictions, strict=True):
        draw_profile(ax, np.ones(len(VALUES), dtype=bool))
        drawstyle = "steps-mid" if method == "nearest" else "default"
        ax.plot(profile_positions, result[:-1], drawstyle=drawstyle, color=SELECTED, linewidth=1.5, zorder=2)
        ax.scatter(TARGET_POSITION, result[-1], color=TARGET, s=42, edgecolor="white", linewidth=0.7, zorder=5)
        draw_method_title(ax, method, supports_points)
        ax.set_ylim(lower, upper)

    handles = [
        Line2D([], [], marker="o", linestyle="none", color=SELECTED, label="Observations", markersize=4),
        Line2D([], [], color=SELECTED, label="Resampled profile", linewidth=1),
        Line2D([], [], marker="o", linestyle="none", color=TARGET, label="Result at target", markersize=5),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=3,
        frameon=False,
        fontsize=9,
        handletextpad=0.5,
    )
    _add_input_legend(fig)
    fig.subplots_adjust(left=0.04, right=0.985, top=0.85, bottom=0.15, hspace=1.5, wspace=0.30)
    return fig


def make_reducer_diagram() -> plt.Figure:
    """
    Summarize the same five selected cells with every built-in reducer.

    We use equal weights and select one raster row with GridNeighbours. draw_profile() shows which source cells
    contribute. Counts and distances are labeled separately because their units differ from the measured values.
    """

    # Use one row of five cells for every reducer
    raster = gu.Raster.from_array(np.tile(VALUES, (9, 1)), Affine(1, 0, 0, 0, -1, 9), 32631, nodata=-9999)
    window = GridNeighbours(offsets=tuple((0, offset) for offset in range(-2, 3)))
    operators: list[tuple[str, Reducer, str]] = [
        ("mean", Mean(neighborhood=window), ""),
        ("sum", Sum(neighborhood=window), ""),
        ("median", Median(neighborhood=window), ""),
        ("quantile", Quantile(0.6, neighborhood=window), ""),
        ("q1", Quantile(0.25, neighborhood=window), ""),
        ("q3", Quantile(0.75, neighborhood=window), ""),
        ("mode", Mode(neighborhood=window), ""),
        ("minimum", Minimum(neighborhood=window), ""),
        ("maximum", Maximum(neighborhood=window), ""),
        ("range", Range(neighborhood=window), ""),
        ("rms", RootMeanSquare(neighborhood=window), ""),
        ("std", StandardDeviation(neighborhood=window), ""),
        ("count", Count(neighborhood=window), "observations"),
        ("average_distance", AverageDistance(neighborhood=window), "pixels"),
        ("average_distance_pts", AveragePairwiseDistance(neighborhood=window), "pixels"),
    ]
    locations = np.array([TARGET_POSITION]), np.array([4.5])

    # Calculate results first so all panels can share a value scale
    results = []
    for method, operator, unit in operators:
        result = np.asarray(raster.resample_at_points(locations, operator, as_array=True)).item()
        results.append((method, result, unit))
    largest_value = max(result for _, result, unit in results if not unit)
    upper = max(VALUES.max(), largest_value) + 1.5
    selected = (POSITIONS >= 1.5) & (POSITIONS <= 5.5)
    fig, axes = plt.subplots(4, 4, figsize=(8.6, 7.6), sharex=True, sharey=True)

    # Shade the selected window and mark each result at the target
    for ax, (method, result, unit) in zip(axes.flat, results):
        ax.axvspan(1, 6, ymax=0.85, facecolor=SELECTED, alpha=0.16, zorder=0)
        draw_profile(ax, selected)
        if not unit:
            ax.hlines(result, 1, 6, color=TARGET, linewidth=1.2, zorder=2)
            ax.scatter(TARGET_POSITION, result, color=TARGET, s=42, edgecolor="white", linewidth=0.7, zorder=4)

        # Counts and distances need labels because they use different units
        if unit:
            value_label = f"{result:.2f}".rstrip("0").rstrip(".")
            ax.text(0.5, 0.80, f"{value_label} {unit}", transform=ax.transAxes, ha="center", fontsize=9, color=NEUTRAL)
        if method == "quantile":
            ax.text(0.5, 0.80, "q=0.6", transform=ax.transAxes, ha="center", fontsize=9, color=NEUTRAL)
        draw_method_title(ax, method, supports_points=True)
        ax.set_ylim(-0.5, upper)

    axes.flat[-1].set_axis_off()
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=SELECTED, label="Selected observations", markersize=5),
        Line2D([], [], marker="o", linestyle="none", color=FADED, label="Outside the window", markersize=5),
        Line2D([], [], marker="o", linestyle="none", color=TARGET, label="Result at target", markersize=5),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=3,
        frameon=False,
        fontsize=9,
        handletextpad=0.5,
    )
    _add_input_legend(fig)
    fig.subplots_adjust(left=0.04, right=0.985, top=0.88, bottom=0.145, hspace=1.6, wspace=0.30)

    # Center the three panels in the last row
    spacing = axes.flat[1].get_position().x0 - axes.flat[0].get_position().x0
    for ax in axes.flat[12:15]:
        bounds = ax.get_position()
        ax.set_position((bounds.x0 + spacing / 2, bounds.y0, bounds.width, bounds.height))
    return fig
