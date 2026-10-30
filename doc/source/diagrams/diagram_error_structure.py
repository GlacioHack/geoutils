"""Illustrate systematic bias, random error magnitude and spatial correlation in one error field."""

from __future__ import annotations

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.image import AxesImage
from matplotlib.patches import ConnectionPatch
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

NEUTRAL = "#253246"
UNCERTAINTY = "#4C78A8"
PAIR_COLORS = ("#3975a7", "#D97732", "#39835c")


def draw_error_field(
    ax: plt.Axes,
    field: NDArray[np.float64],
    extent: float,
    point: NDArray[np.float64],
    neighbours: NDArray[np.float64],
) -> AxesImage:
    """Show one error realization and three separation distances from point A, in metres."""

    # Center the color scale on zero so the offset remains visible
    limit = float(np.max(np.abs(field)))
    image = ax.imshow(field, origin="lower", extent=(0, extent, 0, extent), cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set(xlabel="X (m)", ylabel="Y (m)", title="One random error field")
    ax.set_xticks([0, 300, 600])
    ax.set_yticks([0, 300, 600])
    ax.set_anchor("N")

    # Match each point pair to its color on the correlation curve
    for number, (neighbour, color) in enumerate(zip(neighbours, PAIR_COLORS, strict=True), 1):
        arrow = ax.annotate(
            "",
            xy=neighbour,
            xytext=point,
            arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.8, "shrinkA": 5, "shrinkB": 5},
        )
        assert arrow.arrow_patch is not None
        arrow.arrow_patch.set_path_effects(
            [path_effects.Stroke(linewidth=4, foreground="white"), path_effects.Normal()]
        )
        ax.scatter(*neighbour, s=35, facecolor="white", edgecolor=color, linewidth=1.5, zorder=4)

        # Offset distance labels from the arrow shafts
        separation = neighbour - point
        perpendicular = np.array([-separation[1], separation[0]]) / np.linalg.norm(separation)
        label_position = (neighbour + point) / 2 + 16 * perpendicular
        ax.text(
            *label_position,
            f"$d_{number}$",
            color=color,
            ha="center",
            va="center",
            fontsize=12,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1},
        )

    # Identify the point whose distribution is shown on the right
    ax.scatter(*point, s=55, facecolor=NEUTRAL, edgecolor="white", linewidth=1.3, zorder=5)
    ax.annotate(
        "A",
        point,
        xytext=(7, 7),
        textcoords="offset points",
        ha="left",
        va="bottom",
        color=NEUTRAL,
        fontsize=14,
        weight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1},
    )
    return image


def draw_error_distribution(ax: plt.Axes, true_value: float, bias: float, magnitude: float, point_error: float) -> None:
    """Compare an estimate with the true value and the distribution of possible estimates at point A."""

    # Show the marginal distribution rather than pooling different spatial locations
    mean_estimate = true_value + bias
    estimate_limits = (mean_estimate - 3.5 * magnitude, mean_estimate + 3.5 * magnitude)
    estimates = np.linspace(*estimate_limits, 500)
    density = norm.pdf(estimates, loc=mean_estimate, scale=magnitude)
    ax.plot(estimates, density, color=NEUTRAL, linewidth=1.6)
    within_uncertainty = np.abs(estimates - mean_estimate) <= magnitude
    ax.fill_between(estimates, 0, density, where=within_uncertainty, color=UNCERTAINTY, alpha=0.3)
    ax.set(
        xlim=estimate_limits,
        ylim=(0, 0.92),
        ylabel="Probability density (1/m)",
        title="Distribution of estimates at A\n(across possible realizations)",
    )
    ax.set_xticks(true_value + np.arange(-2, 4))
    ax.set_yticks([0, 0.2, 0.4])
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    # Measure bias from the true value and random magnitude from the mean
    ax.axvline(true_value, color="0.55", linestyle=":", linewidth=1.1, ymax=0.74)
    ax.axvline(mean_estimate, color=NEUTRAL, linestyle="--", linewidth=1, ymax=0.74)
    ax.annotate(
        "",
        xy=(mean_estimate, 0.67),
        xytext=(true_value, 0.67),
        arrowprops={"arrowstyle": "<->", "color": NEUTRAL, "mutation_scale": 8, "shrinkA": 0, "shrinkB": 0},
    )
    ax.text(true_value + bias / 2, 0.7, "Systematic error\n(bias)", ha="center", va="bottom", fontsize=11)

    # Illustrate uncertainty with a one-standard-deviation interval
    ax.annotate(
        "",
        xy=(mean_estimate + magnitude, 0.55),
        xytext=(mean_estimate, 0.55),
        arrowprops={"arrowstyle": "<->", "color": UNCERTAINTY},
    )
    ax.text(
        mean_estimate + magnitude + 0.25,
        0.55,
        "Random error\nmagnitude σ\n(often called\n" r"$\bf{uncertainty}$)",
        ha="left",
        va="center",
        fontsize=11,
        color=UNCERTAINTY,
    )

    # Show the estimate corresponding to the error at A
    point_estimate = true_value + point_error
    point_density = norm.pdf(point_estimate, loc=mean_estimate, scale=magnitude)
    ax.scatter(point_estimate, point_density, s=35, color=NEUTRAL, edgecolor="white", zorder=4)
    ax.annotate(
        "Estimate",
        xy=(point_estimate, point_density),
        xytext=(0.03, 0.21),
        textcoords="axes fraction",
        ha="left",
        va="center",
        fontsize=11,
        weight="bold",
        arrowprops={"arrowstyle": "->", "color": NEUTRAL, "lw": 0.9},
    )

    # Measure the estimate's error from the true value
    error_midpoint = (point_estimate + true_value) / 2
    ax.annotate(
        "",
        xy=(true_value, point_density),
        xytext=(point_estimate, point_density),
        arrowprops={"arrowstyle": "<->", "color": NEUTRAL, "lw": 1.2, "shrinkA": 5, "shrinkB": 0},
    )
    ax.annotate(
        r"$\bf{Error}$",
        xy=(error_midpoint, point_density),
        xytext=(error_midpoint, 0.055),
        ha="center",
        va="center",
        fontsize=11,
        arrowprops={"arrowstyle": "->", "color": NEUTRAL, "lw": 0.9},
    )

    # Mark the true value at zero error
    ax.text(
        true_value - 0.12,
        0.61,
        r"$\bf{True\ value}$" "\n(unknown)",
        ha="right",
        va="center",
        fontsize=11,
        color="0.4",
    )
    ax.set_xlabel("Estimate (m)")


def draw_error_correlation(
    ax: plt.Axes,
    distances: NDArray[np.float64],
    correlation_ranges: NDArray[np.float64],
    variance_fractions: NDArray[np.float64],
) -> None:
    """Match distances on the error field to correlations with two spatial ranges."""

    # Weight each correlation curve by its share of the total error variance
    lags = np.linspace(0, 400, 500)
    scaled_lags = lags[:, None] / correlation_ranges
    contributions = variance_fractions * np.exp(-4 * scaled_lags**2)
    correlation = contributions.sum(axis=1)
    ax.plot(lags, correlation, color=NEUTRAL, linewidth=1.6)
    ax.set(
        xlim=(0, 400),
        ylim=(0, 1),
        ylabel="Error correlation",
        title="Error correlation between points\n(averaged over the domain)",
    )
    ax.set_xticks([0, 100, 200, 300, 400])
    ax.set_yticks([0, 0.5, 1], labels=["0%", "50%", "100%"])
    ax.grid(axis="y", color="0.9", linewidth=0.6)

    # Use the same color and distance as each arrow on the map
    scaled_distances = distances[:, None] / correlation_ranges
    pair_correlations = (variance_fractions * np.exp(-4 * scaled_distances**2)).sum(axis=1)
    transform = ax.get_xaxis_transform()
    for number, (distance, value, color) in enumerate(zip(distances, pair_correlations, PAIR_COLORS, strict=True), 1):
        ax.vlines(distance, 0, value, color=color, linewidth=1, linestyles=":")
        ax.scatter(distance, value, s=35, color=color, edgecolor="white", zorder=4)
        label = f"$d_{number}$: {value:.0%}"
        if number == 1:
            label = r"$\bf{Correlation}$ at " + label
        ax.annotate(
            label,
            (distance, value),
            xytext=(-6, 8) if number == len(distances) else (6, 8),
            textcoords="offset points",
            ha="right" if number == len(distances) else "left",
            color=color,
            fontsize=11,
        )

        # Repeat each map distance below the correlation axis
        height = -0.36 - 0.12 * (number - 1)
        ax.annotate(
            "",
            xy=(0, height),
            xytext=(distance, height),
            xycoords=transform,
            arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.5},
            annotation_clip=False,
            clip_on=False,
        )
        ax.text(
            distance / 2, height + 0.025, f"$d_{number}$", transform=transform, ha="center", va="bottom", color=color
        )
    ax.set_xlabel("Distance between points (m)")


def make_error_structure_diagram() -> plt.Figure:
    """
    Draw a spatial error field, possible estimates at one point, and error correlation between locations.

    draw_error_field() marks point A and three pairs. draw_error_distribution() compares an estimate at A with the
    true value. draw_error_correlation() shows how the same pairs correlate at their different distances.
    """

    # Generate a Gaussian field with a constant bias and known pointwise standard deviation
    true_value = 100.0
    bias = 0.25
    magnitude = 0.8
    extent = 600.0
    resolution = 3.0
    correlation_ranges = np.array([80.0, 400.0])
    variance_fractions = np.array([0.5, 0.5])
    size = int(extent / resolution)
    random = np.random.default_rng(42)
    field = np.full((size, size), bias)

    # Gaussian smoothing gives an effective range of four kernel widths
    for correlation_range, fraction in zip(correlation_ranges, variance_fractions, strict=True):
        smoothing = correlation_range / (4 * resolution)
        padding = int(4 * smoothing + 0.5)
        white_noise = random.normal(size=(size + 2 * padding, size + 2 * padding))
        smoothed = gaussian_filter(white_noise, sigma=smoothing)

        # A separable 2D filter has standard deviation sum(kernel**2) for unit white noise
        offsets = np.arange(-padding, padding + 1)
        kernel = np.exp(-0.5 * (offsets / smoothing) ** 2)
        kernel /= kernel.sum()
        component = smoothed[padding:-padding, padding:-padding] / np.sum(kernel**2)

        # Independent component variances add to the total variance
        field += magnitude * np.sqrt(fraction) * component

    # Land the map connection on the distribution's left side
    point = np.array([540.0, 480.0])
    neighbours = np.array([[540.0, 420.0], [420.0, 360.0], [190.0, 480.0]])
    distances = np.linalg.norm(neighbours - point, axis=1)
    point_error = float(field[int(point[1] / resolution), int(point[0] / resolution)])

    # Show one realization, one point across repeats, and the dependence between point pairs
    with plt.rc_context(
        {"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12}
    ):
        # Align both columns with the square map
        figure = plt.figure(figsize=(12, 6.9))
        grid = figure.add_gridspec(2, 2, width_ratios=[1.25, 1])
        figure.subplots_adjust(left=0.055, right=0.93, bottom=0.24, top=0.93, wspace=0.45, hspace=0.6)
        field_axis = figure.add_subplot(grid[:, 0])
        distribution_axis = figure.add_subplot(grid[0, 1])
        correlation_axis = figure.add_subplot(grid[1, 1])
        image = draw_error_field(field_axis, field, extent, point, neighbours)
        draw_error_distribution(distribution_axis, true_value, bias, magnitude, point_error)
        draw_error_correlation(correlation_axis, distances, correlation_ranges, variance_fractions)

        # Link the error at A to the same realization on its distribution
        connection = ConnectionPatch(
            xyA=point,
            xyB=(true_value + point_error, norm.pdf(point_error, loc=bias, scale=magnitude)),
            coordsA=field_axis.transData,
            coordsB=distribution_axis.transData,
            color=NEUTRAL,
            linewidth=1.1,
            linestyle="--",
            connectionstyle="angle3,angleA=0,angleB=90",
            shrinkA=5,
            shrinkB=5,
            zorder=3,
        )
        figure.add_artist(connection)

        # Label the map without competing with the point-pair annotations
        colorbar_axis = field_axis.inset_axes((0.1, -0.15, 0.8, 0.035))
        colorbar = figure.colorbar(image, cax=colorbar_axis, orientation="horizontal")
        colorbar.set_label("Error (m)")
    return figure


fig = make_error_structure_diagram()
plt.show()
