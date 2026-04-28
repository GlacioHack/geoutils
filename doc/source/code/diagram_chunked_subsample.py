"""Script to make diagram for chunked polygonize in documentation."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from geoutils.raster.array import get_mask_from_array
from geoutils.stats.sampling import (
    _get_subsample_size_from_user_input,
    _splitmix64,
    _subsample_numpy,
)


# -----------------------------------------------------------------------------
# Example data
# -----------------------------------------------------------------------------

ARR = np.array(
    [
        [1.2, 2.1, 2.8, 3.4, 4.0, np.nan, np.nan, 1.1],
        [0.9, 1.5, 2.7, 3.1, 4.2, np.nan, np.nan, 0.6],
        [0.8, 1.4, 2.2, 2.9, 3.9, 4.1, np.nan, 1.2],
        [0.7, 1.0, np.nan, np.nan, 3.0, 3.4, 2.0, 1.5],
        [1.6, 1.9, np.nan, np.nan, 2.6, 2.9, 1.4, 1.0],
        [2.2, 1.1, 1.7, 4.2, 3.8, 3.1, 2.1, 1.7],
    ],
    dtype=np.float32,
)

NROWS, NCOLS = ARR.shape
CHUNK_ROWS = (3, 3)
CHUNK_COLS = (4, 4)

USER_SUBSAMPLE = 0.25
SEED = 7

DISPLAY_K = 5
EXAMPLE_CHUNK = (0, 1)  # Top-right chunk

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------

NEUTRAL = "#333333"
PIXEL_GRID = "0.80"
CHUNK_COLOR = "#222222"
VALID_COLOR = "#6BAED6"
VALID_EDGE = "#4C78A8"
INVALID_COLOR = "#F7F7F7"
HIGHLIGHT_COLOR = "#F58518"
TOPK_COLOR = "#4C78A8"
SEQUENTIAL_COLOR = "#F58518"


# -----------------------------------------------------------------------------
# Helpers for subsampling and valid values per chunk
# -----------------------------------------------------------------------------

def _first_valid_gids_in_chunk(arr: np.ndarray, chunk_loc: tuple[int, int], n: int = 5) -> np.ndarray:
    """Return the first n valid global linear indices in one chunk, in flattened local order."""
    r0, r1, c0, c1 = _chunk_bounds(*chunk_loc)
    block = arr[r0:r1, c0:c1]
    mask = ~get_mask_from_array(block)

    flat_local = np.flatnonzero(mask.ravel())[:n]
    if flat_local.size == 0:
        return np.array([], dtype=np.int64)

    rr = flat_local // block.shape[1] + r0
    cc = flat_local % block.shape[1] + c0
    return np.ravel_multi_index((rr, cc), arr.shape).astype(np.int64)

def _chunk_bounds(iy: int, ix: int) -> tuple[int, int, int, int]:
    """Return raster chunk bounds as (row0, row1, col0, col1)."""
    row_starts = np.concatenate([[0], np.cumsum(CHUNK_ROWS)])
    col_starts = np.concatenate([[0], np.cumsum(CHUNK_COLS)])
    return row_starts[iy], row_starts[iy + 1], col_starts[ix], col_starts[ix + 1]


def _chunk_valid_counts(arr: np.ndarray) -> list[int]:
    """Valid counts in row-major chunk order."""
    counts: list[int] = []
    for iy in range(len(CHUNK_ROWS)):
        for ix in range(len(CHUNK_COLS)):
            r0, r1, c0, c1 = _chunk_bounds(iy, ix)
            block = arr[r0:r1, c0:c1]
            counts.append(int(np.count_nonzero(~get_mask_from_array(block))))
    return counts


def _global_valid_indices(arr: np.ndarray) -> np.ndarray:
    """Global linear indices of valid pixels."""
    return np.flatnonzero(~get_mask_from_array(arr).ravel())


def _topk_choice_with_keys(arr: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return top-k gids and keys using the package SplitMix64 implementation."""
    valids = _global_valid_indices(arr).astype(np.uint64)
    keys = _splitmix64(np.uint64(seed) ^ valids)
    sel = np.argpartition(keys, k - 1)[:k]
    sel = sel[np.lexsort((valids[sel], keys[sel]))]
    return valids[sel].astype(np.int64), keys[sel]


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------

def _setup_axis(ax: plt.Axes, *, xlim: tuple[float, float], ylim: tuple[float, float], equal: bool = True) -> None:
    """Common axis formatting."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal" if equal else "auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_raster_cells(ax: plt.Axes, arr: np.ndarray) -> None:
    """Draw valid / invalid raster cells."""
    mask = get_mask_from_array(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            x = j
            y = arr.shape[0] - 1 - i
            is_valid = not bool(mask[i, j])

            rect = Rectangle(
                (x, y),
                1.0,
                1.0,
                facecolor=VALID_COLOR if is_valid else INVALID_COLOR,
                edgecolor="none",
                alpha=0.25 if is_valid else 1.0,
                zorder=1,
            )
            ax.add_patch(rect)

            if is_valid:
                ax.text(
                    x + 0.5,
                    y + 0.5,
                    f"{arr[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=NEUTRAL,
                    zorder=3,
                )


def _draw_pixel_grid(ax: plt.Axes, *, nrows: int, ncols: int, extend: float = 0.0) -> None:
    """Draw thin pixel grid."""
    segments: list[np.ndarray] = []

    for x in range(ncols + 1):
        segments.append(np.array([[x, -extend], [x, nrows + extend]], dtype=float))
    for y in range(nrows + 1):
        segments.append(np.array([[-extend, y], [ncols + extend, y]], dtype=float))

    ax.add_collection(
        LineCollection(
            segments,
            colors=PIXEL_GRID,
            linewidths=0.8,
            capstyle="round",
            joinstyle="round",
            zorder=2,
            clip_on=False,
        )
    )


def _draw_chunk_boundaries(ax: plt.Axes, *, nrows: int, ncols: int, extend: float = 0.0) -> None:
    """Draw chunk boundaries."""
    row_starts = np.concatenate([[0], np.cumsum(CHUNK_ROWS)])
    col_starts = np.concatenate([[0], np.cumsum(CHUNK_COLS)])

    for x in col_starts:
        ax.plot(
            [x, x],
            [-extend, nrows + extend],
            color=CHUNK_COLOR,
            linewidth=2.3,
            zorder=4,
            solid_capstyle="round",
            clip_on=False,
        )

    for y_idx in row_starts:
        y = nrows - y_idx
        ax.plot(
            [-extend, ncols + extend],
            [y, y],
            color=CHUNK_COLOR,
            linewidth=2.3,
            zorder=4,
            solid_capstyle="round",
            clip_on=False,
        )


def _draw_example_count_chunk(ax: plt.Axes, chunk_loc: tuple[int, int], *, color: str = HIGHLIGHT_COLOR) -> None:
    """Highlight one example chunk used for counting valid pixels."""
    r0, r1, c0, c1 = _chunk_bounds(*chunk_loc)

    left = c0
    right = c1
    top = NROWS - r0
    bottom = NROWS - r1

    rect = Rectangle(
        (left, bottom),
        right - left,
        top - bottom,
        fill=False,
        edgecolor=color,
        linewidth=4.0,
        zorder=7,
        joinstyle="round",
    )
    ax.add_patch(rect)


def _draw_chunk_count_labels(
    ax: plt.Axes,
    counts: list[int],
    *,
    example_chunk: tuple[int, int] | None = None,
) -> None:
    """Draw per-chunk valid counts, with one example chunk highlighted."""
    k = 0
    for iy in range(len(CHUNK_ROWS)):
        for ix in range(len(CHUNK_COLS)):
            r0, r1, c0, c1 = _chunk_bounds(iy, ix)
            x = 0.5 * (c0 + c1)
            y = NROWS - 0.5 * (r0 + r1)

            is_example = example_chunk is not None and (iy, ix) == example_chunk

            ax.text(
                x,
                y - 1.1,
                f"n = {counts[k]}",
                ha="center",
                va="center",
                fontsize=10,
                color=HIGHLIGHT_COLOR if is_example else NEUTRAL,
                fontweight="bold" if is_example else None,
                bbox=dict(
                    boxstyle="round,pad=0.20",
                    fc="white",
                    ec=HIGHLIGHT_COLOR if is_example else "0.85",
                    lw=1.0 if is_example else 0.8,
                    alpha=0.95,
                ),
                zorder=6,
            )
            k += 1

def _draw_topk_selected(ax: plt.Axes, arr: np.ndarray, chosen_gids: np.ndarray) -> None:
    """Draw topk-selected pixels as orange dots."""
    for gid in chosen_gids:
        i, j = np.unravel_index(int(gid), arr.shape)
        x = j + 0.5
        y = arr.shape[0] - 1 - i + 0.5

        ax.scatter(
            x,
            y,
            s=60,
            facecolor=HIGHLIGHT_COLOR,
            edgecolor="white",
            linewidth=1.0,
            zorder=7,
        )


def _add_title(fig: plt.Figure, title: str) -> None:
    """Global grey title."""
    fig.text(
        0.5,
        0.99,
        title,
        ha="center",
        va="top",
        fontsize=15,
        fontweight="semibold",
        color="0.35",
    )


def _add_visual_legend(ax: plt.Axes) -> None:
    """Draw a compact single-column legend in the upper-right free space of the top panel."""
    x_sym0 = 0.95
    x_sym1 = 1.0
    x_txt = 1.03

    y0 = 0.68
    dy = 0.1

    entries = [
        ("pixel_grid", "Pixel grid"),
        ("chunk_boundary", "Chunk boundary"),
        ("valid", "Valid pixel"),
        ("invalid", "Invalid / nodata"),
        ("example", "Illustrated example pixel"),
        ("selected", "Selected sample pixel"),
    ]

    for i, (kind, label) in enumerate(entries):
        y = y0 - i * dy

        if kind == "pixel_grid":
            ax.plot(
                [x_sym0, x_sym1],
                [y, y],
                transform=ax.transAxes,
                color=PIXEL_GRID,
                lw=0.8,
                solid_capstyle="round",
                clip_on=False,
            )

        elif kind == "chunk_boundary":
            ax.plot(
                [x_sym0, x_sym1],
                [y, y],
                transform=ax.transAxes,
                color=CHUNK_COLOR,
                lw=2.3,
                solid_capstyle="round",
                clip_on=False,
            )

        elif kind == "valid":
            rect = Rectangle(
                (x_sym0, y - 0.018),
                x_sym1 - x_sym0,
                0.036,
                transform=ax.transAxes,
                facecolor=VALID_COLOR,
                edgecolor=VALID_EDGE,
                linewidth=1.0,
                alpha=0.25,
                clip_on=False,
            )
            ax.add_patch(rect)

        elif kind == "invalid":
            rect = Rectangle(
                (x_sym0, y - 0.018),
                x_sym1 - x_sym0,
                0.036,
                transform=ax.transAxes,
                facecolor=INVALID_COLOR,
                edgecolor="0.75",
                linewidth=1.0,
                clip_on=False,
            )
            ax.add_patch(rect)

        elif kind == "example":
            ax.scatter(
                [0.5 * (x_sym0 + x_sym1)],
                [y],
                transform=ax.transAxes,
                s=45,
                facecolor=TOPK_COLOR,
                edgecolor="white",
                linewidth=1.0,
                zorder=20,
                clip_on=False,
            )

        elif kind == "selected":
            ax.scatter(
                [0.5 * (x_sym0 + x_sym1)],
                [y],
                transform=ax.transAxes,
                s=45,
                facecolor=HIGHLIGHT_COLOR,
                edgecolor="white",
                linewidth=1.0,
                zorder=20,
                clip_on=False,
            )

        ax.text(
            x_txt,
            y,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=9.5,
            color=NEUTRAL,
        )

def _valid_rank_map(arr: np.ndarray) -> dict[int, int]:
    """Map global linear index -> rank in flattened valid-value order."""
    valids = _global_valid_indices(arr).astype(np.int64)
    return {int(gid): i for i, gid in enumerate(valids)}

def _draw_sequential_selected(ax: plt.Axes, arr: np.ndarray, chosen_gids: np.ndarray) -> None:
    """Draw sequentially selected pixels as orange dots."""
    for gid in chosen_gids:
        i, j = np.unravel_index(int(gid), arr.shape)
        x = j + 0.5
        y = arr.shape[0] - 1 - i + 0.5

        ax.scatter(
            x,
            y,
            s=60,
            facecolor=HIGHLIGHT_COLOR,
            edgecolor="white",
            linewidth=1.0,
            zorder=7,
        )

def _draw_topk_example_labels(
    ax: plt.Axes,
    arr: np.ndarray,
    example_gids: np.ndarray,
    *,
    seed: int,
    selected_gids: np.ndarray,
) -> None:
    """Label example pixels with deterministic topk keys using log10(key)."""
    gids_u64 = example_gids.astype(np.uint64)
    keys = _splitmix64(np.uint64(seed) ^ gids_u64)
    selected_set = set(selected_gids.astype(np.int64).tolist())

    for gid, key in zip(example_gids, keys, strict=True):
        i, j = np.unravel_index(int(gid), arr.shape)
        x = j + 0.5
        y = arr.shape[0] - 1 - i + 0.5

        display_val = int(key >> 56)
        is_selected = int(gid) in selected_set
        color = HIGHLIGHT_COLOR if is_selected else TOPK_COLOR

        ax.text(
            x,
            y + 0.48,
            f"k={display_val}",
            ha="center",
            va="bottom",
            fontsize=7.6,
            color=color,
            bbox=dict(
                boxstyle="round,pad=0.15",
                fc="white",
                ec="0.85",
                lw=0.8,
                alpha=0.96,
            ),
            zorder=8,
        )

def _sequential_chunk_order_rank_map(arr: np.ndarray) -> dict[int, int]:
    """
    Map global linear index to rank in flattened valid order by chunk.

    Ordering is:
    - chunks in row-major order
    - within each chunk, valid pixels in flattened local order
    """
    rank_map: dict[int, int] = {}
    rank = 1

    for iy in range(len(CHUNK_ROWS)):
        for ix in range(len(CHUNK_COLS)):
            r0, r1, c0, c1 = _chunk_bounds(iy, ix)
            block = arr[r0:r1, c0:c1]
            mask = ~get_mask_from_array(block)

            flat_local = np.flatnonzero(mask.ravel())
            ncols = block.shape[1]

            for flat in flat_local:
                rr = r0 + flat // ncols
                cc = c0 + flat % ncols
                gid = int(np.ravel_multi_index((rr, cc), arr.shape))
                rank_map[gid] = rank
                rank += 1

    return rank_map

def _draw_sequential_example_labels(
    ax: plt.Axes,
    arr: np.ndarray,
    example_gids: np.ndarray,
    *,
    selected_gids: np.ndarray,
) -> None:
    """Label example pixels with their rank in chunk-wise flattened valid order."""
    rank_map = _sequential_chunk_order_rank_map(arr)
    selected_set = set(selected_gids.astype(np.int64).tolist())

    for gid in example_gids:
        i, j = np.unravel_index(int(gid), arr.shape)
        x = j + 0.5
        y = arr.shape[0] - 1 - i + 0.5

        is_selected = int(gid) in selected_set
        color = HIGHLIGHT_COLOR if is_selected else TOPK_COLOR

        ax.text(
            x,
            y + 0.48,
            f"n={rank_map[int(gid)]}",
            ha="center",
            va="bottom",
            fontsize=8.0,
            color=color,
            bbox=dict(
                boxstyle="round,pad=0.15",
                fc="white",
                ec="0.85",
                lw=0.8,
                alpha=0.96,
            ),
            zorder=8,
        )

def _example_valid_gids_sparse(arr: np.ndarray) -> np.ndarray:
    """
    Return example valid gids through the first row and into the middle of the
    second row, keeping one valid pixel out of two for readability.
    """
    valids = _global_valid_indices(arr).astype(np.int64)

    row_break_1 = CHUNK_ROWS[0] * NCOLS
    row_break_2 = (CHUNK_ROWS[0] + CHUNK_ROWS[1] // 2) * NCOLS

    gids = valids[valids < row_break_2]
    return gids[::2]

def _draw_example_reference_dots(
    ax: plt.Axes,
    arr: np.ndarray,
    example_gids: np.ndarray,
    *,
    selected_gids: np.ndarray,
) -> None:
    """Draw blue dots for illustrated example pixels that are not selected."""
    selected_set = set(selected_gids.astype(np.int64).tolist())

    for gid in example_gids:
        if int(gid) in selected_set:
            continue

        i, j = np.unravel_index(int(gid), arr.shape)
        x = j + 0.5
        y = arr.shape[0] - 1 - i + 0.5

        ax.scatter(
            x,
            y,
            s=60,
            facecolor=TOPK_COLOR,
            edgecolor="white",
            linewidth=1.0,
            zorder=6.5,
        )

def _lowest_key_example_gids(example_gids: np.ndarray, *, seed: int, n_low: int = 2) -> np.ndarray:
    """Return gids of the n_low smallest keys among example gids."""
    gids_u64 = example_gids.astype(np.uint64)
    keys = _splitmix64(np.uint64(seed) ^ gids_u64)

    m = min(n_low, len(example_gids))
    sel = np.argpartition(keys, m - 1)[:m]
    return example_gids[sel].astype(np.int64)

# -----------------------------------------------------------------------------
# Main figure
# -----------------------------------------------------------------------------

def make_chunked_subsample_diagram() -> tuple[plt.Figure, np.ndarray]:
    """Build a 3-panel schematic for chunked subsampling."""

    EXAMPLE_GIDS = _example_valid_gids_sparse(ARR)
    LOW_KEY_EXAMPLE_GIDS = _lowest_key_example_gids(EXAMPLE_GIDS, seed=SEED, n_low=2)

    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.28, wspace=0.28)

    ax_top = fig.add_subplot(gs[0, :])
    ax_bl = fig.add_subplot(gs[1, 0])
    ax_br = fig.add_subplot(gs[1, 1])

    axes = np.array([ax_top, ax_bl, ax_br], dtype=object)

    valid_counts = _chunk_valid_counts(ARR)
    total_valids = int(np.count_nonzero(~get_mask_from_array(ARR)))
    k = _get_subsample_size_from_user_input(USER_SUBSAMPLE, total_valids)

    # Use package subsampling function for both strategies
    sequential_gids_rc = _subsample_numpy(
        ARR,
        subsample=USER_SUBSAMPLE,
        return_indices=True,
        random_state=SEED,
        strategy="sequential",
    )
    sequential_gids = np.ravel_multi_index(sequential_gids_rc, ARR.shape).astype(np.int64)

    topk_gids_rc = _subsample_numpy(
        ARR,
        subsample=USER_SUBSAMPLE,
        return_indices=True,
        random_state=SEED,
        strategy="topk",
    )
    topk_gids = np.ravel_multi_index(topk_gids_rc, ARR.shape).astype(np.int64)

    # Keys only for labeling the topk panel
    topk_gids_for_keys, topk_keys = _topk_choice_with_keys(ARR, k, SEED)

    r0, r1, c0, c1 = _chunk_bounds(*EXAMPLE_CHUNK)
    example_count = int(np.count_nonzero(~get_mask_from_array(ARR[r0:r1, c0:c1])))

    # Top panel
    _draw_raster_cells(ax_top, ARR)
    _draw_pixel_grid(ax_top, nrows=NROWS, ncols=NCOLS)
    _draw_chunk_boundaries(ax_top, nrows=NROWS, ncols=NCOLS)
    _draw_chunk_count_labels(ax_top, valid_counts, example_chunk=EXAMPLE_CHUNK)
    _draw_example_count_chunk(ax_top, EXAMPLE_CHUNK)

    ax_top.text(
        0.5,
        1.0,
        "Count valid values per chunk",
        transform=ax_top.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=NEUTRAL,
    )

    ax_top.text(
        -0.36,
        0.50,
        "We sum per-chunk\nvalid values to get\nrequested subsample size\nfrom total valid count\n"
        "(as it may be a fraction\n or exceed total count)",
        transform=ax_top.transAxes,
        ha="center",
        va="center",
        fontsize=10.2,
        color=NEUTRAL,
        linespacing=1.3,
    )

    _add_visual_legend(ax_top)
    _setup_axis(ax_top, xlim=(-0.5, NCOLS + 2.8), ylim=(-0.5, NROWS + 0.5))

    # Bottom left: topk
    _draw_raster_cells(ax_bl, ARR)
    _draw_pixel_grid(ax_bl, nrows=NROWS, ncols=NCOLS)
    _draw_chunk_boundaries(ax_bl, nrows=NROWS, ncols=NCOLS)
    _draw_example_count_chunk(ax_bl, (0, 0), color="0.55")

    _draw_example_reference_dots(ax_bl, ARR, EXAMPLE_GIDS, selected_gids=topk_gids)
    _draw_topk_selected(ax_bl, ARR, topk_gids)
    _draw_topk_example_labels(
        ax_bl,
        ARR,
        EXAMPLE_GIDS,
        seed=SEED,
        selected_gids=LOW_KEY_EXAMPLE_GIDS
    )

    ax_bl.text(
        0.5,
        1.0,
        "topk (chunk-invariant)",
        transform=ax_bl.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=NEUTRAL,
    )
    ax_bl.text(
        0.5,
        -0.02,
        "Each valid pixel gets a deterministic uint64 key\nfrom its row/col indexes (rounded rank shown)\n"
        "the k smallest keys are selected",
        transform=ax_bl.transAxes,
        ha="center",
        va="top",
        fontsize=10.0,
        color=NEUTRAL,
        linespacing=1.25,
    )
    _setup_axis(ax_bl, xlim=(-0.5, NCOLS + 0.5), ylim=(-0.5, NROWS + 0.5))

    # Bottom right: sequential
    _draw_raster_cells(ax_br, ARR)
    _draw_pixel_grid(ax_br, nrows=NROWS, ncols=NCOLS)
    _draw_chunk_boundaries(ax_br, nrows=NROWS, ncols=NCOLS)
    _draw_example_count_chunk(ax_br, (0, 0), color="0.55")

    _draw_example_reference_dots(ax_br, ARR, EXAMPLE_GIDS, selected_gids=sequential_gids)
    _draw_sequential_selected(ax_br, ARR, sequential_gids)
    _draw_sequential_example_labels(ax_br, ARR, EXAMPLE_GIDS, selected_gids=sequential_gids)

    ax_br.text(
        0.5,
        1.0,
        "sequential (chunk-dependent)",
        transform=ax_br.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=NEUTRAL,
    )

    ax_br.text(
        0.5,
        -0.02,
        "A random draw is applied to flattened valid indexes,\n"
        "fast selection but depends on chunking",
        transform=ax_br.transAxes,
        ha="center",
        va="top",
        fontsize=10.0,
        color=NEUTRAL,
        linespacing=1.25,
    )

    _setup_axis(ax_br, xlim=(-0.5, NCOLS + 0.5), ylim=(-0.5, NROWS + 0.5))

    _add_title(fig, "Chunked subsampling of valid raster values")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.13)
    return fig, axes


fig, _ = make_chunked_subsample_diagram()
plt.show()
