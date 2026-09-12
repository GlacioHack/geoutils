# Copyright (c) 2025 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Apply global/grouped statistics and variography to rasters and point clouds."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Hashable, Iterable, Mapping
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from geoutils._dispatch import _is_pointcloud, _is_raster
from geoutils._misc import import_optional
from geoutils._typing import ArrayLike, NDArrayNum
from geoutils.stats.grouping import _grouped_stats, _validate_group_declarations
from geoutils.stats.reduction import _normalize_statistics, _reduce_global_values
from geoutils.stats.selection import (
    _sample_and_mask_global_values,
    _select_values_and_mask_at_support,
)
from geoutils.stats.variography import Variogram, _estimate_variogram

if TYPE_CHECKING:
    from geoutils.interface.interpolation import InterpolationMethod
    from geoutils.multiproc import MultiprocConfig
    from geoutils.pointcloud.base import PointCloudBase
    from geoutils.pointcloud.pointcloud import PointCloudLike
    from geoutils.raster.base import RasterBase, RasterLike
    from geoutils.stats.reduction import _Statistics
    from geoutils.vector.base import VectorLike

__all__ = ["stats", "variogram"]


def _global_stats(
    values: ArrayLike | Mapping[str, ArrayLike],
    statistics: _Statistics,
    *,
    mask: Any | None,
    subsample: int | float,
    random_state: int | np.random.Generator | None,
    strategy: Literal["auto", "dense", "sparse", "groupwise"],
    subsampling_strategy: Literal["sequential", "topk"],
    mp_config: MultiprocConfig | None,
) -> Any:
    """
    Calculate global statistic.

    This is done the same way as grouped statistics, simply by creating one implicit group!

    _sample_and_mask_global_values() chooses a common sample and applies the mask.
    _reduce_global_values() reduces every selected value through the same calculation as one complete grouped bin.

    One selected value returns its statistics directly, several values return a dictionary keyed by their names.

    See stats() for argument descriptions.

    :param values: One numeric array or container, or a mapping of output names to selected arrays with matching
        shapes, as returned by _select_values_and_mask_at_support().
    :param mask: Boolean eligibility array on the selected locations, or None to keep every location.

    :returns: Statistics for one selected value, or a mapping from value names to their statistics.
    """

    # Sample the global values accounting for the optional user-input mask and subsample size
    values_to_reduce = _sample_and_mask_global_values(
        values,
        mask=mask,
        subsample=subsample,
        random_state=random_state,
        subsampling_strategy=subsampling_strategy,
        mp_config=mp_config,
    )

    # Reduce all values through one shared calculation and restore the established global output form
    return _reduce_global_values(
        values_to_reduce,
        statistics,
        strategy=strategy,
        mp_config=mp_config if subsample == 1 else None,
    )


def stats(
    source: RasterLike | PointCloudLike | ArrayLike | Mapping[str, ArrayLike],
    statistics: str | Callable[[Any], Any] | Iterable[str | Callable[[Any], Any]] | None = None,
    *,
    by: Mapping[str, Any] | None = None,
    values: int | str | Iterable[int | str] | Mapping[str, Any] | None = None,
    bins: Mapping[str, Any] | None = None,
    categories: Mapping[str, Iterable[Hashable]] | None = None,
    at: Literal["self"] | RasterLike | PointCloudLike | None = None,
    mask: RasterLike | PointCloudLike | VectorLike | ArrayLike | None = None,
    mask_mode: Literal["inside", "outside"] = "inside",
    subsample: int | float = 1,
    subsample_per_group: bool = False,
    random_state: int | np.random.Generator | None = None,
    strategy: Literal["auto", "dense", "sparse", "groupwise"] = "auto",
    backend: Literal["geoutils", "flox"] = "geoutils",
    subsampling_strategy: Literal["sequential", "topk"] = "topk",
    interpolation: InterpolationMethod = "linear",
    align: Literal["raise", "reproject"] = "raise",
    observed: bool = True,
    return_masks: bool = False,
    mp_config: MultiprocConfig | None = None,
) -> Any:
    """
    Calculate statistics, either global (whole array) or grouped with other geospatial objects (continuous binning or
    categorical grouping, including zonal grouping).

    Omit ``by`` for global statistics. With ``by``, provide ``categories`` for discrete groups, ``bins`` for
    continuous groups along the variable. For a vector, provide the feature column through a tuple directly in ``by``,
    which performs geometric zonal statistics::

        # Global statistics
        raster.stats()
        raster.stats("mean")
        raster.stats(["mean", "std", "nmad"])

        # Categorical stats
        raster.stats(["mean", "std"], by={"landcover": lc}, categories={"landcover": lc_classes})
        # Binned stats
        raster.stats(["mean", "std"], by={"elevation": dem}, bins={"elevation": elevation_bins})
        # Zonal stats
        raster.stats(["mean", "std"], by={"feature": (outlines, "id")})

        # Multiple grouping with vector and raster inputs
        raster.stats(["mean", "std"], by={"elevation": dem, "feature": (outlines, "id")},
        bins={"elevation": elevation_bins})

    Use ``values`` with index of band (for raster) or label of column (for point cloud) to select input, which
    defaults to all bands for a raster, and the main data column for a point cloud.

        # Select only band 2 (defaults to all bands)
        raster.stats(["mean", "std"], values=[1, 2])

        # Select specific point cloud columns (defaults to main column)
        point.stats(["mean", "std"], values=["z", "intensity"])

        # Provide name mapping of bands for output dataframe
        raster.stats(["mean", "std"], values={"red": 1, "blue": 2})

    Raster and point clouds are cosampled at the spatial support of ``at``. Set it to "self" to use the source
    locations; otherwise, point inputs take precedence when no support is specified.

    Source input can be masked with ``mask`` and ``mask_mode``, and subsampled using ``subsample`` and
    ``subsample_per_group``.

    Supported statistics
    --------------------

    - Mean: arithmetic mean of the data,
    - Median: middle value when the valid data points are sorted in increasing order,
    - Max: maximum value among the data,
    - Min: minimum value among the data,
    - Sum: sum of all data,
    - Sum of squares: sum of the squares of all data,
    - 90th percentile: point below which 90% of the data falls,
    - IQR (Interquartile Range): difference between the 75th and 25th percentile of a dataset
    - LE90 (Linear Error with 90% confidence): difference between the 95th and 5th percentiles of a dataset, \
    representing the range within which 90% of the data points lie.
    - NMAD (Normalized Median Absolute Deviation): robust measure of variability in the data, less sensitive to \
    outliers compared to standard deviation.
    - RMSE (Root Mean Square Error): commonly used to express the magnitude of errors or variability and can give \
    insight into the spread of the data. Only relevant when the raster represents a difference of two objects. \
    - Std (Standard deviation): measures the spread or dispersion of the data around the mean,
    - Valid count: number of unmasked entries for masked arrays, or finite entries for ordinary arrays.
    - Total count: total size of the raster.
    - Percentage valid points: ratio between Valid count and Total count.

    If an inlier mask is passed:
    - Total inlier count: number of data points in the inlier mask.
    - Valid inlier count: number of unmasked data points in the array after applying the inlier mask.
    - Percentage inlier points: ratio between Valid inlier count and Valid count. Useful for classification statistics.
    - Percentage valid inlier points: ratio between Valid inlier count and Total inlier count.

    Callable functions are supported as well.

    Grouping details
    ----------------

    Every grouper must have an entry in ``bins`` or ``categories`` unless it has a boolean or Pandas categorical
    dtype. Numeric edge sequences use left-closed intervals and include the final right edge. Pass a
    :class:`pandas.IntervalIndex` to control edge closure explicitly. The result index follows the order of ``by``;
    columns have ``value`` and ``statistic`` levels, and a finite ``count`` is always included for each value.

    When ``return_masks`` is true, the second result behaves as a mapping from each dataframe index key to a boolean
    array or spatial object. Its masks describe complete eligible group membership after ``mask`` and valid groupers,
    before random subsampling and independently of missing selected values.

    Chunk strategies
    ----------------
    Dask and multiprocessing divide the inputs into chunks and use the same NumPy calculation on each one. ``dense``
    stores a fixed-size summary containing every declared group per chunk. It is efficient when the number of group
    combinations is moderate. ``sparse`` stores only groups present in each chunk, reducing memory when many possible
    groups are absent, at the cost of aligning group labels when chunks are combined. Both aggregate counts, sums,
    minima, maxima, means, standard deviations and RMSE across chunks without keeping the original values.

    Median, NMAD, other quantiles, and user functions cannot be aggregated across chunks from the ``dense`` or
    ``sparse`` summaries because they need all values from a group at once. Requesting these combinations raises an
    error. ``groupwise`` instead finds the chunks containing each group and collects all its values before calculation,
    so its memory use depends on the largest group. ``auto`` selects ``groupwise`` for these statistics; otherwise, it
    selects ``dense`` for up to 4096 group combinations and ``sparse`` above that. With ordinary in-memory inputs, all
    group values are already available and these statistics are calculated directly.

    Subsampling strategies
    ----------------------
    Subsampling is global by default. Set ``subsample_per_group=True`` to apply the fraction or maximum separately
    to each combination of grouping variables. Fractions round down within each group; a maximum larger than a
    group keeps every eligible location in that group. ``subsample=1`` always keeps all eligible locations.

    ``topk`` assigns a reproducible SplitMix64 key to each flat input position and keeps the smallest keys, giving
    the same sample for NumPy, Dask and multiprocessing regardless of chunk layout. ``sequential`` draws without
    replacement from the available positions; its result can depend on chunk layout.

    Per-group sampling delegates to _stratified_subsample_indices(). Fractions use full group counts before rounding
    quotas; fixed topk caps select directly within chunks. Dask and multiprocessing share the selection calculation;
    Dask combines topk candidates in a task tree, while multiprocessing combines bounded batches in the caller.
    Group counts and the final sample must fit in memory.

    Sampling uses group membership and the user mask, then selects the same locations for every value array.
    Missing observations are still handled separately for each value. Dask computes the selected arrays together;
    the reduced sample is then calculated eagerly for both Dask and multiprocessing inputs. Without subsampling,
    _reduce_values() uses their shared block calculations, with scheduling differences documented in
    _aggregate_chunked().

    References
    ----------
    The overall calculation follows the split-apply-combine strategy described by Wickham (2011),
    "The Split-Apply-Combine Strategy for Data Analysis":
    https://doi.org/10.18637/jss.v040.i01

    The pairwise mean and variance merge follows Chan, Golub and LeVeque (1983),
    "Algorithms for Computing the Sample Variance: Analysis and Recommendations":
    https://doi.org/10.1080/00031305.1983.10483115

    The key mixer used by ``topk`` follows SplitMix64 from Steele, Lea and Flood (2014),
    "Fast Splittable Pseudorandom Number Generators":
    https://doi.org/10.1145/2660193.2660195

    :param source: Input raster or point cloud to summarize or group.
    :param statistics: Statistics to calculate (e.g. "mean", ["mean", "nmad"], or np.nanmedian). None returns
        "min", "max", "mean", "median", "std", "nmad", "validcount", "totalcount" and "percentagevalidpoints".
        "all" also includes "sum", "sumofsquares", "90thpercentile", "iqr", "le90" and "rmse", plus inlier counts
        for masked global statistics. Grouped defaults replace "validcount" with "count"; every grouped result
        includes "count".
    :param by: Named variables to group by (e.g. {"elevation": dem}); use {"glacier": (outlines, "id")}
        for vector zones. Arrays must match the source input shape. Omit for global statistics.
    :param values: Bands (e.g. [1, 3]) or point columns (e.g. "height") to summarize; defaults to all raster bands
        or the main point data column. Use a mapping to name your band inputs (e.g. {"elevation": (dem, 1)}).
    :param bins: Continuous bins keyed by grouping name (e.g. {"elevation": 10}). Each definition is a count of
        equal-width bins, increasing edges (e.g. [0, 2, 5]), or a Pandas IntervalIndex to choose open/closed sides.
    :param categories: Ordered categories keyed by grouping name (e.g. {"landcover": [100, 110, 120]}).
        Values outside these categories are excluded.
    :param at: Grid or ordered point locations on which to calculate statistics (e.g. at=reference or at="self").
        Defaults to the first point input, if present, otherwise the source locations.
    :param mask: Locations to include (True in a boolean mask, e.g. mask=dem > 1000, or features in a vector mask).
        Global counts describe values before this mask; "all" adds counts for values kept by the mask.
    :param mask_mode: Keep locations "inside" or "outside" vector features; ignored for boolean masks.
    :param subsample: Fraction (e.g. 0.1 for 10%) or maximum count (e.g. 10000) of eligible locations to use.
        A value of 1 keeps all locations. Counts describe the sampled locations.
    :param subsample_per_group: Whether to apply subsample within each combined group (True, stratified sampling) or
    once across all groups (False). Without ``by``, both use one global sample.
    :param random_state: Seed to reproduce subsampling (e.g. 42), or an existing random generator.
    :param strategy: Combine chunk statistics for all groups ("dense"), only groups present in each chunk
        ("sparse"), or gather each complete group ("groupwise"). "auto" chooses from the statistics and group count;
        exact quantiles and custom functions require "auto" or "groupwise" for chunked data.
    :param backend: Use the GeoUtils reducer ("geoutils") or optional Flox reducer ("flox") for grouped statistics.
        Flox cannot return group masks, sample within groups, use multiprocessing, or calculate custom functions and
        NMAD. Its Dask path also excludes exact medians and percentiles.
    :param subsampling_strategy: "topk" keeps the same sampled locations across chunk layouts for a fixed seed;
        "sequential" draws random locations using traversal order and can depend on the chunks.
    :param interpolation: Raster values at point locations use interp_points() with SciPy methods "nearest",
        "linear", "slinear", "cubic", "quintic", "pchip" or "splinef2d".
        Raster groupers listed in categories use "nearest".
    :param align: "raise" rejects different grids or coordinate systems; "reproject" aligns them to the output
        locations. Point inputs must still share the same ordered coordinates.
    :param observed: Omit declared group combinations with no eligible locations (True), or include them (False).
    :param return_masks: Also return masks keyed by group labels (e.g. table, masks = raster.stats(...)).
        Masks cover complete groups before subsampling. Requires by.
    :param mp_config: Worker and tile settings for multiprocessing, e.g. MultiprocConfig(chunks=512).
        Cannot be combined with Dask inputs.
    :returns: A statistic, summary dictionary, grouped dataframe, or grouped dataframe and mask mapping.
    """

    # 1/ Validate user inputs
    if strategy not in {"auto", "dense", "sparse", "groupwise"}:
        raise ValueError("Argument ``strategy`` must be 'auto', 'dense', 'sparse' or 'groupwise'.")
    if backend not in {"geoutils", "flox"}:
        raise ValueError("Argument ``backend`` must be 'geoutils' or 'flox'.")
    if subsampling_strategy not in {"sequential", "topk"}:
        raise ValueError("Argument ``subsampling_strategy`` must be 'sequential' or 'topk'.")
    if not isinstance(subsample, (int, float)) or subsample <= 0:
        raise ValueError("Argument ``subsample`` must be a positive number.")
    if not isinstance(subsample_per_group, (bool, np.bool_)):
        raise TypeError("Argument ``subsample_per_group`` must be a boolean.")
    if by is None and (bins is not None or categories is not None or not observed or return_masks):
        raise ValueError(
            "Argument ``by`` is required for ``bins``, ``categories``, ``observed``=False or ``return_masks``=True."
        )
    if backend == "flox":
        if by is None:
            raise ValueError("The Flox backend requires grouped statistics through argument ``by``.")
        if subsample_per_group or return_masks or mp_config is not None or strategy != "auto":
            raise ValueError(
                "The Flox backend requires ``subsample_per_group``=False, ``return_masks``=False, "
                "``mp_config``=None and ``strategy``='auto'."
            )
        import_optional("flox", extra_name="flox")

        # Warn before spatial selection reads values from raster or point cloud containers
        spatial_inputs = [source, at, mask, *by.values(), *(values.values() if isinstance(values, Mapping) else [])]
        spatial_inputs = [value[0] if isinstance(value, tuple) and value else value for value in spatial_inputs]
        if any(_is_raster(value) or _is_pointcloud(value) for value in spatial_inputs):
            warnings.warn(
                "The Flox backend loads Raster and PointCloud inputs before calculating grouped statistics.",
                category=UserWarning,
                stacklevel=2,
            )
    if (
        by is None
        and statistics is not None
        and not isinstance(statistics, (str, Iterable))
        and not callable(statistics)
    ):
        warnings.warn(f"Statistic name {statistics} is a not recognized string", category=UserWarning)
        return None

    # Validate statistics and group inputs
    normalized_statistics = _normalize_statistics(
        statistics, grouped=by is not None, masked=by is None and mask is not None
    )
    definitions = None if by is None else _validate_group_declarations(by, bins, categories)

    # 2/ Dispatch to input preparation and global/grouped statistics subfunctions
    with ExitStack() as stack:
        # Inspect all inputs to choose the common "support" (raster grid or point locations), then reproject
        # the values from source + the mask on it, as those are used by both global/grouped stats
        # (But we don't reproject grouping variables for now, this is done in _grouped_stats below)
        values_at_support, support_mask, support = _select_values_and_mask_at_support(
            source,
            by=by,
            values=values,
            at=at,
            mask=mask,
            mask_mode=mask_mode,
            interpolation=interpolation,
            align=align,
            mp_config=mp_config,
            stack=stack,
        )

        # If no grouping, compute global stats
        if by is None:
            return _global_stats(
                values_at_support,
                normalized_statistics,
                mask=support_mask,
                subsample=subsample,
                random_state=random_state,
                strategy=strategy,
                subsampling_strategy=subsampling_strategy,
                mp_config=mp_config,
            )

        # Otherwise perform grouping
        assert definitions is not None
        return _grouped_stats(
            source,
            by,
            values=values_at_support,
            support=support,
            statistics=normalized_statistics,
            mask=support_mask,
            subsample=subsample,
            subsample_per_group=subsample_per_group,
            random_state=random_state,
            strategy=strategy,
            backend=backend,
            subsampling_strategy=subsampling_strategy,
            interpolation=interpolation,
            align=align,
            observed=observed,
            return_masks=return_masks,
            mp_config=mp_config,
            definitions=definitions,
            stack=stack,
        )


def variogram(
    source: RasterBase | PointCloudBase,
    *,
    band: int | None = None,
    n_pairs: int = 1_000_000,
    sampling: Literal["loglag", "random_xy"] = "loglag",
    estimator: str | Callable[[NDArrayNum], float] = "dowd",
    bins: Literal["log", "uniform"] | Iterable[float] = "log",
    n_lags: int = 24,
    min_lag: float | None = None,
    max_lag: float | None = None,
    n_runs: int = 1,
    model: str | Callable[..., Any] | list[str | Callable[..., Any]] | None = None,
    fit_kwargs: dict[str, Any] | None = None,
    random_state: int | np.random.Generator | None = None,
    mask: RasterLike | VectorLike | ArrayLike | None = None,
    **pair_sampling_kwargs: Any,
) -> Variogram:
    """
    Estimate a variogram from spatial pairs efficiently sampled from a raster or point cloud.

    The output Variogram object can be easily re-fit with ``fit()``, plotted with ``plot()`` or exported
    to various formats (e.g., GSTools, GPyTorch) with for example ``to_gstools()``. It can also be passed to
    resampling and gridding function supporting kriging in GeoUtils.

    The empirical variogram's ``sampling`` uses pairwise logarithmic distance "loglag" by default to efficiently
    capture both short distances and large distances on large datasets, which typically outperforms random
    endpoints selection ("random_xy").
    Both options support out-of-memory execution through Dask for raster inputs.

    SciKit-GStat supplies the semivariance estimators and theoretical model formulas. GeoUtils fits the chosen
    model to the measured distance bins with SciPy curve_fit().

    :param source: Raster or point cloud whose spatial variability is estimated.
    :param band: Raster band to sample, counting from 1. Omit for a point cloud.
    :param n_pairs: Number of finite pairs targeted in each run (e.g. 100_000).
    :param sampling: How to select pairs: ``"loglag"`` balances short and long distances, while ``"random_xy"``
        selects endpoints independently.
    :param estimator: Semivariance estimator from SciKit-GStat: ``"dowd"``, ``"matheron"``, ``"cressie"``,
        ``"genton"``, ``"minmax"``, ``"entropy"`` or ``"percentile"``. A function can instead map absolute pair
        differences to one value per distance bin.
    :param bins: Distance bins: ``"log"`` for logarithmic spacing, ``"uniform"`` for equal widths, or explicit
        edges (e.g. [1, 10, 100]).
    :param n_lags: Number of distance bins when bins is ``"log"`` or ``"uniform"``.
    :param min_lag: Minimum sampled distance in CRS units; defaults to the smaller pixel spacing or half the estimated
        point spacing.
    :param max_lag: Maximum sampled distance in CRS units; defaults to the diagonal of the source locations.
    :param n_runs: Independent samples to average; repeat sampling to estimate each distance bin's standard error.
    :param model: SciKit-GStat model to fit: ``"spherical"``, ``"exponential"``, ``"gaussian"``, ``"cubic"``,
        ``"stable"`` or ``"matern"``, or the corresponding model function. Sum a list of models ordered from short
        to long range (e.g. ["spherical", "exponential"]). ``None`` keeps only the empirical variogram.
    :param fit_kwargs: Options for Variogram.fit(): ``use_nugget``, ``bounds``, ``p0`` or ``maxfev``
        (e.g. {"use_nugget": True}); optimization uses SciPy curve_fit().
    :param random_state: Seed or NumPy generator for reproducible sampling across runs (e.g. 42).
    :param mask: Locations to keep: True values in a boolean mask, an aligned mask raster for raster inputs,
        or locations inside vector geometries.
    :param pair_sampling_kwargs: Extra pairsample() options for the source (e.g. ``strategy`` or ``max_rounds``).
    :returns: Variogram with distance bins, pair counts and semivariance, plus sampling errors and a fitted model
        when requested.
    """

    # Keep raster band selection out of point pair sampling kwargs
    pair_kwargs: dict[str, Any] = {} if band is None else {"band": band}
    pair_kwargs.update(
        {
            "n_pairs": n_pairs,
            "sampling": sampling,
            "min_distance": min_lag,
            "max_distance": max_lag,
            "mask": mask,
        }
    )
    pair_kwargs.update(pair_sampling_kwargs)

    return _estimate_variogram(
        source,
        n_runs=n_runs,
        estimator=estimator,
        bins=bins,
        n_lags=n_lags,
        min_lag=min_lag,
        max_lag=max_lag,
        models=model,
        fit_kwargs=fit_kwargs,
        random_state=random_state,
        pair_kwargs=pair_kwargs,
    )
