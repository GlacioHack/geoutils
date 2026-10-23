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

"""Apply global/grouped statistics to rasters and point clouds."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Hashable, Iterable, Mapping
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from geoutils._dispatch import _is_pointcloud, _is_raster, is_dask_array
from geoutils._misc import import_optional
from geoutils._typing import ArrayLike
from geoutils.operators.base import LocalData
from geoutils.operators.overlap import OverlapBackend
from geoutils.operators.reducer import Reducer
from geoutils.raster.array import get_mask_from_array
from geoutils.stats.grouping import _grouped_stats, _validate_group_declarations
from geoutils.stats.reduction import (
    _check_input_flox_engine,
    _check_input_reduction_method,
    _check_input_tdigest,
    _normalize_statistics,
    _reduce_global_values,
)
from geoutils.stats.selection import (
    _coordinates_at_support,
    _sample_and_mask_global_values,
    _select_values_and_mask_at_support,
)

if TYPE_CHECKING:
    from geoutils.interface.resampling import InterpolationMethod
    from geoutils.multiproc import MultiprocConfig
    from geoutils.pointcloud.base import PointCloudBase
    from geoutils.pointcloud.pointcloud import PointCloudLike
    from geoutils.raster.base import RasterBase, RasterLike
    from geoutils.stats.reduction import _Statistics
    from geoutils.uncertainty import ErrorStructure, PropagationSummary
    from geoutils.vector.base import VectorLike

__all__ = ["stats"]


def _global_stats(
    values: ArrayLike | Mapping[str, ArrayLike],
    statistics: _Statistics,
    *,
    mask: Any | None,
    subsample: int | float,
    random_state: int | np.random.Generator | None,
    subsampling_strategy: Literal["sequential", "topk"],
    mp_config: MultiprocConfig | None,
    support: RasterBase | PointCloudBase | None = None,
    _return_local_data: bool = False,
    backend: str = "geoutils",
    flox_engine: str | None = None,
    tdigest_compression: float = 100,
    method: str = "auto",
    reindex: bool | None = None,
    split_every: int | None = None,
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
    result = _reduce_global_values(
        values_to_reduce,
        statistics,
        mp_config=mp_config if subsample == 1 else None,
        backend=backend,
        flox_engine=flox_engine,
        tdigest_compression=tdigest_compression,
        method=method,
        reindex=reindex,
        split_every=split_every,
    )
    if not _return_local_data:
        return result

    # Use original array positions as source IDs so repeated selections receive the same source error
    coordinates = _coordinates_at_support(support)
    local_inputs = []
    names = list(values_to_reduce)
    for name in names:
        array = values_to_reduce[name][0]
        if is_dask_array(array) or not isinstance(array, np.ndarray):
            raise ValueError("Statistics uncertainty propagation currently requires in-memory values.")
        flat_values = np.asanyarray(np.ma.getdata(array)).reshape(-1)
        valid = ~get_mask_from_array(array).reshape(-1)
        source_ids = np.empty(array.size, dtype=object)
        source_ids[:] = [(name, position) for position in range(array.size)]
        local_inputs.append(
            LocalData(
                values=flat_values,
                valid=valid,
                source_ids=source_ids,
                coordinates=coordinates,
            )
        )
    nominal = np.asarray([result] if len(names) == 1 else [result[name] for name in names], dtype=float)
    return result, local_inputs, names, nominal


def stats(
    source: RasterLike | PointCloudLike | ArrayLike | Mapping[str, ArrayLike],
    statistics: str | Callable[[Any], Any] | Reducer | Iterable[str | Callable[[Any], Any] | Reducer] | None = None,
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
    method: Literal["auto", "map-reduce", "blockwise", "cohorts", "groupwise"] = "auto",
    reindex: bool | None = None,
    split_every: int | None = None,
    backend: Literal["geoutils", "flox"] = "geoutils",
    flox_engine: str | None = None,
    tdigest_compression: float = 100,
    subsampling_strategy: Literal["sequential", "topk"] = "topk",
    interpolation: InterpolationMethod = "linear",
    align: Literal["raise", "reproject"] = "raise",
    observed: bool = True,
    return_masks: bool = False,
    mp_config: MultiprocConfig | None = None,
    fractional: bool = False,
    overlap_backend: OverlapBackend = "auto",
    error_structure: ErrorStructure | None = None,
    uncertainty_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    """
    Calculate statistics, either global (whole array) or grouped with other geospatial objects (continuous binning or
    categorical grouping, including zonal grouping).

    Supports Dask or multiprocessing to compute statistics per chunk on large rasters or point clouds.

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
    defaults to all bands for a raster, and the main data column for a point cloud::

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
    - IQR (Interquartile Range): difference between the 75th and 25th percentile of a dataset.
    - LE90 (Linear Error with 90% confidence): difference between the 95th and 5th percentiles of a dataset,
      representing the range within which 90% of the data points lie.
    - NMAD (Normalized Median Absolute Deviation): robust measure of variability in the data, less sensitive to
      outliers compared to standard deviation.
    - RMSE (Root Mean Square Error): commonly used to express the magnitude of errors or variability and can give
      insight into the spread of the data. Only relevant when the raster represents a difference of two objects.
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

    With both Flox and GeoUtils backend, "map-reduce" is the default, which maps groups in each chunk, reduces them,
    then merges results. This leads to exact results for mergeable statistics (e.g. mean, STD) but approximate for
    non-mergeable ones (e.g. median, NMAD).

    GeoUtils backend also supports "groupwise", which collects all values within a group. It still reads chunk by
    chunk the large input data, never loading it in memory, but its memory usage will be proportional to the largest
    group. This method allows to derive exact statistics for any reducer.

    Subsampling strategies
    ----------------------
    Subsampling is global by default. Set ``subsample_per_group=True`` to apply it within each grouping variables, for
    instance to limit large groups to maximum 1,000,000 points. Combined with ``method="groupwise"``, this allows to
    derive exact statistics on all small groups, and approximate ones (due to subsampling) on large groups, capping
    maximum memory naturally.

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
        or the main point data attribute. Use a mapping to name your band inputs (e.g. {"elevation": (dem, 1)}).
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
    :param method: Parallel calculation: "auto", "map-reduce", GeoUtils-only "groupwise", or Flox-only
        "blockwise" and "cohorts". Auto uses mergeable summaries where available, including t-digests for quantiles.
    :param reindex: Include every declared group in chunk summaries (True), only groups present (False), or let
        the backend choose (None). Groupwise does not use summaries. Flox cohorts and t-digests require False or None.
    :param split_every: Maximum block summaries per merge (at least 2). None uses Dask configuration or the backend
        default (8 for GeoUtils).
    :param backend: Use the GeoUtils reducer ("geoutils") or optional Flox reducer ("flox") for global or
        grouped statistics. Both support in-memory and Dask inputs; multiprocessing requires GeoUtils. Exact quantiles
        use complete groups; map-reduce uses t-digests for quantiles. Group masks and custom functions require
        GeoUtils.
    :param flox_engine: Local Flox reduction algorithm: None (automatic), "numpy", "numba", "flox" or "numbagg".
        This is independent of Dask/MP execution. The "numbagg" engine does not support t-digests.
    :param tdigest_compression: Digest accuracy/memory setting from 20 to 1000 (default 100).
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
    :param fractional: Weight source raster cells by the exact area covered by one vector grouping. Overlapping feature
        groups are calculated independently.
    :param overlap_backend: Library used to calculate polygon coverage of raster cells. Auto uses ExactExtract when it
        is installed and the grid is compatible, and otherwise uses Shapely.
    :param error_structure: Optional source error model for one supplied Reducer. When supplied, return the statistic
        result and a PropagationSummary calculated from the same selected values.
    :param uncertainty_kwargs: Options passed to geoutils.uncertainty.propagate().
    :returns: A statistic, summary dictionary, grouped dataframe, or grouped dataframe and mask mapping.
    """

    # 1/ Validate user inputs
    if backend not in {"geoutils", "flox"}:
        raise ValueError("Argument ``backend`` must be 'geoutils' or 'flox'.")
    _check_input_reduction_method(backend, method, reindex, split_every)
    if fractional and (method not in {"auto", "map-reduce"} or reindex is not None or split_every is not None):
        raise ValueError(
            "Fractional statistics require method='auto' or 'map-reduce', reindex=None and split_every=None."
        )
    if backend == "flox" and mp_config is not None:
        raise ValueError("The Flox backend does not support multiprocessing.")
    _check_input_flox_engine(backend, flox_engine)
    _check_input_tdigest(tdigest_compression)
    if subsampling_strategy not in {"sequential", "topk"}:
        raise ValueError("Argument ``subsampling_strategy`` must be 'sequential' or 'topk'.")
    if not isinstance(subsample, (int, float)) or subsample <= 0:
        raise ValueError("Argument ``subsample`` must be a positive number.")
    if not isinstance(subsample_per_group, (bool, np.bool_)):
        raise TypeError("Argument ``subsample_per_group`` must be a boolean.")
    if not isinstance(fractional, (bool, np.bool_)):
        raise TypeError("Argument ``fractional`` must be a boolean.")
    if fractional:
        if by is None:
            raise ValueError("Fractional statistics require one vector grouper through argument by.")
        if subsample != 1 or subsample_per_group:
            raise ValueError("Fractional statistics do not support subsampling.")
        if return_masks:
            raise ValueError("Fractional statistics require return_masks=False.")
    if error_structure is not None:
        if not isinstance(statistics, Reducer):
            raise TypeError("Statistics uncertainty propagation requires one Reducer as statistics.")
        if subsample != 1:
            raise ValueError("Statistics uncertainty propagation does not yet support subsampling.")
        if backend != "geoutils":
            raise ValueError("Statistics uncertainty propagation requires backend='geoutils'.")
    if by is None and (bins is not None or categories is not None or not observed or return_masks):
        raise ValueError(
            "Argument ``by`` is required for ``bins``, ``categories``, ``observed``=False or ``return_masks``=True."
        )
    if backend == "flox":
        if return_masks:
            raise ValueError("The Flox backend requires ``return_masks``=False.")
        import_optional("flox")

        # Warn before spatial selection reads values from raster or point cloud containers
        spatial_inputs = [
            source,
            at,
            mask,
            *(by.values() if by is not None else []),
            *(values.values() if isinstance(values, Mapping) else []),
        ]
        spatial_inputs = [value[0] if isinstance(value, tuple) and value else value for value in spatial_inputs]
        if (
            not fractional
            and mp_config is None
            and any(
                (_is_raster(value) and not value._is_xr) or (_is_pointcloud(value) and not value._is_dask)
                for value in spatial_inputs
            )
        ):
            warnings.warn(
                "The Flox backend loads Raster and PointCloud inputs before calculating grouped statistics.",
                category=UserWarning,
                stacklevel=2,
            )
    if (
        by is None
        and statistics is not None
        and not isinstance(statistics, (str, Iterable, Reducer))
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
            result = _global_stats(
                values_at_support,
                normalized_statistics,
                mask=support_mask,
                subsample=subsample,
                random_state=random_state,
                subsampling_strategy=subsampling_strategy,
                mp_config=mp_config,
                support=support,
                _return_local_data=error_structure is not None,
                backend=backend,
                flox_engine=flox_engine,
                tdigest_compression=tdigest_compression,
                method=method,
                reindex=reindex,
                split_every=split_every,
            )
        else:
            # For in-memory data, build the group IDs and reduce each selected value directly
            assert definitions is not None
            if fractional:
                if support is None or not _is_raster(support):
                    raise TypeError("Fractional vector statistics require a raster support grid.")
                from geoutils.stats.fractional import _fractional_vector_stats

                result = _fractional_vector_stats(
                    source,
                    by,
                    values=values_at_support,
                    support=cast("RasterBase", support),
                    statistics=normalized_statistics,
                    mask=support_mask,
                    definitions=definitions,
                    observed=observed,
                    overlap_backend=overlap_backend,
                    backend=backend,
                    flox_engine=flox_engine,
                    mp_config=mp_config,
                    return_local_data=error_structure is not None,
                )
            else:
                result = _grouped_stats(
                    source,
                    by,
                    values=values_at_support,
                    support=support,
                    statistics=normalized_statistics,
                    mask=support_mask,
                    subsample=subsample,
                    subsample_per_group=subsample_per_group,
                    random_state=random_state,
                    backend=backend,
                    flox_engine=flox_engine,
                    tdigest_compression=tdigest_compression,
                    method=method,
                    reindex=reindex,
                    split_every=split_every,
                    subsampling_strategy=subsampling_strategy,
                    interpolation=interpolation,
                    align=align,
                    observed=observed,
                    return_masks=return_masks,
                    mp_config=mp_config,
                    definitions=definitions,
                    stack=stack,
                    _return_local_data=error_structure is not None,
                )

    if error_structure is None:
        return result

    # Calculate uncertainty from the same selected values and their original array positions
    nominal_result, local_inputs, output_labels, nominal_values = result
    options = dict(uncertainty_kwargs or {})
    for reserved in ("nominal_estimate", "output_labels"):
        if reserved in options:
            raise ValueError(f"uncertainty_kwargs cannot override {reserved}.")
    from geoutils.uncertainty import propagate

    summary: PropagationSummary = propagate(
        cast(Reducer, statistics),
        local_inputs,
        error_structure,
        nominal_estimate=nominal_values,
        output_labels=output_labels,
        **options,
    )
    return nominal_result, summary
