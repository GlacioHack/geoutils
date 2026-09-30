# Copyright (c) 2026 GeoUtils developers
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

"""
Functionalities for transformations of raster objects.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from copy import copy
from dataclasses import replace
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

import affine
import numpy as np
import rasterio as rio
import shapely
from numpy.typing import NDArray
from packaging.version import Version
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from shapely.geometry import box
from shapely.strtree import STRtree

from geoutils import projtools
from geoutils._config import config
from geoutils._dispatch import _check_crs, _check_match_bbox, _check_match_grid, _clip_geodataframe
from geoutils._misc import import_optional, silence_rasterio_message
from geoutils._typing import DTypeLike, MArrayNum, NDArrayBool, NDArrayNum
from geoutils.interface.rasterization import (
    _normalize_burn_values,
    _partition_burn_by_geogrids,
    _rasterize_selected_on_geogrid,
    _VectorBurnSpec,
)
from geoutils.interface.resampling import _interp_points_base
from geoutils.multiproc.chunked import (
    ChunkedGeoGrid,
    GeoGrid,
    normalize_chunks,
)
from geoutils.multiproc.mparray import (
    MultiprocConfig,
    _split_chunk_size,
    _write_multiproc_result,
)
from geoutils.operators.execution import _evaluate_operator_batch, _resample_at_points
from geoutils.operators.interpolator import Interpolator, Linear, Nearest, _regular_interpolation_method
from geoutils.operators.neighbours import (
    GridCoverage,
    GridNeighbours,
    PointNeighbours,
    _check_regular_grid_neighbours,
    _configure_grid_neighbours,
    _resolve_grid_neighbours_for_interpolator,
)
from geoutils.operators.nodata import NodataHandling, NodataPropagation, _validate_nodata_propagation
from geoutils.operators.overlap import (
    GridIntersection,
    OverlapBackend,
    _corners_are_grid_aligned,
    _exactextract_operation,
    _grid_intersection_fractions_from_corners,
    _grid_intersection_local_data,
    _run_exactextract,
)
from geoutils.operators.reducer import (
    Maximum,
    Mean,
    Minimum,
    Reducer,
    RegularReductionMethod,
    Sum,
    _reduce_overlap_batch,
)
from geoutils.raster.referencing import (
    _default_nodata,
    _ij2xy,
    _res,
    _xy2ij,
)

if TYPE_CHECKING:
    from geoutils.raster.base import RasterLike, RasterType
    from geoutils.raster.raster import Raster
    from geoutils.vector.vector import Vector, VectorLike

# Dask as optional dependency
try:
    import dask.array as da
    from dask import delayed
except ImportError:
    da = None

    def delayed(*args: Any, **kwargs: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Fake delayed decorator if dask is not installed
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            return func

        return decorator


#####################################
# 1/ DOWNSAMPLE AND OVERVIEW OPENING
#####################################


def _overview_level_for_downsample(source: rio.io.DatasetReader, downsample: float) -> int | None:
    """Return the closest suitable stored overview level for a downsampling factor."""

    # Match GDAL's nearest-neighbor overview selection by allowing up to 20% oversampling
    overview_level = None
    for level, overview_factor in enumerate(source.overviews(1)):
        if overview_factor > downsample * 1.2:
            break
        overview_level = level
    return overview_level


@contextmanager
def _open_downsampled_raster(source: rio.io.DatasetReader, downsample: float) -> Iterator[WarpedVRT]:
    """Open a reduced grid, using the closest suitable stored overview when one is available."""

    # Define the requested grid from the native raster rather than from the selected overview
    width = max(1, int(np.floor(source.width / downsample)))
    height = max(1, int(np.floor(source.height / downsample)))
    transform = source.transform * affine.Affine.scale(downsample)

    # Keep the selected overview open while the virtual raster resamples it to the requested grid
    with ExitStack() as stack:
        read_source = source
        overview_level = _overview_level_for_downsample(source, downsample)
        if overview_level is not None:
            read_source = stack.enter_context(rio.open(source.name, overview_level=overview_level))
        vrt = stack.enter_context(
            WarpedVRT(
                read_source,
                crs=source.crs,
                transform=transform,
                width=width,
                height=height,
                resampling=Resampling.nearest,
            )
        )
        yield vrt


###############
# REPROJECT
###############


def _resampling_method_from_str(method_str: str) -> rio.enums.Resampling:
    """Get a rasterio resampling method from a string representation, e.g. "cubic_spline"."""
    # Try to match the string version of the resampling method with a rio Resampling enum name
    for method in rio.enums.Resampling:
        if method.name == method_str:
            resampling_method = method
            break
    # If no match was found, raise an error.
    else:
        raise ValueError(
            f"'{method_str}' is not a valid rasterio.enums.Resampling method. "
            f"Valid methods: {[method.name for method in rio.enums.Resampling]}"
        )
    return resampling_method


def _check_reproj_nodata_dtype(
    source_raster: RasterType,
    nodata: int | float | None,
    dtype: DTypeLike | None,
    force_source_nodata: int | float | None,
) -> tuple[DTypeLike, int | float | None, int | float | None]:
    """Check user inputs of reproject regarding nodata and data type."""

    # Set output dtype
    if dtype is None:
        # Warning: this will not work for multiple bands with different dtypes
        dtype = source_raster.dtype

    # --- Set source nodata if provided -- #
    if force_source_nodata is None:
        src_nodata = source_raster.nodata
    else:
        src_nodata = force_source_nodata
        # Raise warning if a different nodata value exists for this raster than the forced one (not None)
        if source_raster.nodata is not None:
            warnings.warn(
                "Forcing source nodata value of {} despite an existing nodata value of {} in the raster. "
                "To silence this warning, use self.set_nodata() before reprojection instead of forcing.".format(
                    force_source_nodata, source_raster.nodata
                )
            )

    # --- Set destination nodata if provided -- #
    # This is needed in areas not covered by the input data.
    # If None, will use GeoUtils' default, as rasterio's default is unknown, hence cannot be handled properly.
    if nodata is None:
        nodata = source_raster.nodata
        if nodata is None:
            nodata = _default_nodata(dtype)
            # If nodata is already being used, raise a warning.
            if not source_raster.is_loaded:
                warnings.warn(
                    f"For reprojection, nodata must be set. Setting default nodata to {nodata}. You may "
                    f"set a different nodata with `nodata`."
                )

            elif nodata in source_raster.data:
                warnings.warn(
                    f"For reprojection, nodata must be set. Default chosen value {nodata} exists in "
                    f"self.data. This may have unexpected consequences. Consider setting a different nodata with "
                    f"self.set_nodata()."
                )

    return dtype, src_nodata, nodata


def _is_reproj_needed(src_shape: tuple[int, int], reproj_kwargs: dict[str, Any]) -> bool:
    """Check if reprojection is actually needed based on transformation parameters."""

    src_transform = reproj_kwargs["src_transform"]
    transform = reproj_kwargs["dst_transform"]
    src_crs = reproj_kwargs["src_crs"]
    crs = reproj_kwargs["dst_crs"]
    grid_size = reproj_kwargs["dst_shape"][::-1]
    src_res = _res(src_transform)
    res = _res(transform)

    # Caution, grid_size is (width, height) while shape is (height, width)
    return all(
        [
            (transform == src_transform) or (transform is None),
            (crs == src_crs) or (crs is None),
            (grid_size == src_shape[::-1]) or (grid_size is None),
            np.all(np.array(res) == src_res) or (res is None),
        ]
    )


# 2.1/ REPROJECT AN ARRAY WITH RASTERIO
#######################################


def _rio_reproject(src_arr: NDArrayNum, reproj_kwargs: dict[str, Any]) -> NDArrayNum:
    """
    Rasterio reprojection wrapper.

    :param src_arr: Source array for data.
    :param reproj_kwargs: Reprojection parameter dictionary.
    """

    # All masked values must be set to a nodata value for rasterio's reproject to work properly
    if np.ma.isMaskedArray(src_arr):
        is_input_masked = True
        src_mask = np.ma.getmaskarray(src_arr)
        src_arr = src_arr.data  # type: ignore
    else:
        is_input_masked = False
        src_mask = ~np.isfinite(src_arr)

    # Check reprojection is possible with nodata (boolean raster will be converted, so no need to check)
    if np.dtype(src_arr.dtype) != bool and (reproj_kwargs["src_nodata"] is None and np.sum(src_mask) > 0):
        raise ValueError(
            "No nodata set, set one for the raster with self.set_nodata() or use a temporary one "
            "with `force_source_nodata`."
        )

    # For a boolean type
    convert_bool = False
    if np.dtype(src_arr.dtype) == np.bool_:
        # To convert back later
        convert_bool = True
        # Convert to uint8 for nearest, float otherwise
        if reproj_kwargs["resampling"] in [Resampling.nearest, "nearest"]:
            src_arr = src_arr.astype("uint8")  # type: ignore
        else:
            warnings.warn(
                "Reprojecting a raster mask (boolean type) with a resampling method other than 'nearest', "
                "results in the boolean array being converted to float during reprojection."
            )
            src_arr = src_arr.astype("float32")  # type: ignore

        # Convert automated output dtype to the input dtype
        if np.dtype(reproj_kwargs["dtype"]) == np.bool_:
            reproj_kwargs["dtype"] = src_arr.dtype

        # Update nodata value, which won't exist
        reproj_kwargs["src_nodata"] = _default_nodata(src_arr.dtype)

    # Fill with nodata values on mask
    if reproj_kwargs["src_nodata"] is not None:
        src_arr[src_mask] = reproj_kwargs["src_nodata"]

    # Check if multiband
    is_multiband = len(src_arr.shape) > 2

    # Prepare destination array
    shape = (src_arr.shape[0], *reproj_kwargs["dst_shape"]) if is_multiband else reproj_kwargs["dst_shape"]
    dst_arr = np.zeros(shape, dtype=reproj_kwargs["dtype"])

    # Performance keywords
    if reproj_kwargs["num_threads"] == 0:
        # Default to cpu count minus one. If the cpu count is undefined, num_threads will be 1
        cpu_count = os.cpu_count() or 2
        num_threads = cpu_count - 1
    else:
        num_threads = reproj_kwargs["num_threads"]

    # We force XSCALE=1 and YSCALE=1 passed to GDAL.Warp to avoid resampling deformations depending on extent/shape,
    # which leads to different results on chunks or a full array
    # See: https://gdal.org/en/stable/api/gdalwarp_cpp.html#_CPPv415GDALWarpOptions
    # And: https://github.com/rasterio/rasterio/issues/2995
    reproj_kwargs.update(
        {
            "num_threads": num_threads,
            "XSCALE": 1,
            "YSCALE": 1,
        }
    )
    # If Rasterio is recent enough version, force tolerance to 0 to avoid deformations on chunks
    # See: https://github.com/rasterio/rasterio/issues/2433#issuecomment-2786157846
    if Version(rio.__version__) >= Version("1.5.0"):
        reproj_kwargs.update({"tolerance": 0})

    # Pop dtype and dst_shape arguments that don't exist in Rasterio, and are only used above
    reproj_kwargs.pop("dtype")
    reproj_kwargs.pop("dst_shape")

    # Rasterio raises a warning that src_transform are not defined when multiple ones are passed during chunked ops
    # (Dask/Multiproc), although this is not the case, maybe a bug upstream?
    warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)

    # XSCALE/YSCALE have been supported for a while, but not officially exposed in the API until Rasterio 1.5,
    # so we need to silence them in warnings to avoid noise for users
    with silence_rasterio_message(param_name="SCALE"):
        # Run reprojection
        _ = rio.warp.reproject(src_arr, dst_arr, **reproj_kwargs)

    # Get output mask
    if reproj_kwargs["dst_nodata"] is not None:
        dst_mask = dst_arr == reproj_kwargs["dst_nodata"]
    else:
        dst_mask = np.zeros(dst_arr.shape, dtype=bool)

    # If output needs to be converted back to boolean
    if convert_bool:
        dst_arr = dst_arr.astype(bool)

    # Set mask
    if is_input_masked:
        dst_arr = np.ma.masked_array(data=dst_arr, mask=dst_mask, fill_value=reproj_kwargs["dst_nodata"])
    else:
        dst_arr[dst_mask] = np.nan

    return dst_arr


# 2.2 REPROJECT AN ARRAY WITH INDEPENDENT METHODS
#################################################


def _destination_grid_in_source_crs(
    dst_transform: rio.transform.Affine,
    dst_shape: tuple[int, int],
    dst_crs: rio.crs.CRS,
    src_crs: rio.crs.CRS,
    *,
    include_corners: bool,
) -> tuple[NDArrayNum, NDArrayNum, NDArrayNum | None]:
    """
    Return destination centers and, when requested, pixel corners transformed back into the source CRS.

    Resampling is done on the input regular grid for speed, after exact CRS transformation of the center/corner points.

    The two center arrays have length (height * width), ordered by destination row then column.
    Corners have shape (height * width, 4, 2), ordered upper left, upper right, lower right, lower left within
    each cell, or are None when only centers are needed.
    Reducers without a fixed neighborhood use these corners to locate the destination cell in the source grid.
    """

    # Use physical cell centers and boundaries, regardless of the raster's Area/Point interpretation
    rows, cols = np.indices(dst_shape)
    center_x, center_y = _ij2xy(
        rows.reshape(-1),
        cols.reshape(-1),
        transform=dst_transform,
        area_or_point=None,
        shift_area_or_point=False,
        force_offset="center",
    )
    centers = np.column_stack((center_x, center_y))

    # PyProj transforms the above coordinates in X/Y order
    if dst_crs != src_crs:
        transformed_centers = projtools.reproject_points(
            (centers[:, 0], centers[:, 1]), in_crs=dst_crs, out_crs=src_crs
        )
        centers = np.column_stack(transformed_centers)

    # Interpolators and explicit reducer neighborhoods only need the destination centers
    if not include_corners:
        return centers[:, 0], centers[:, 1], None

    # Group the four corners of each destination pixel so a Reducer can find all overlapping source cells
    corner_columns = np.stack((cols, cols + 1, cols + 1, cols), axis=-1).reshape(-1)
    corner_rows = np.stack((rows, rows, rows + 1, rows + 1), axis=-1).reshape(-1)
    corner_x, corner_y = _ij2xy(
        corner_rows,
        corner_columns,
        transform=dst_transform,
        area_or_point=None,
        shift_area_or_point=False,
        force_offset="ul",
    )
    corners = np.column_stack((np.asarray(corner_x), np.asarray(corner_y))).reshape(-1, 4, 2)

    # Transform the boundaries only for reductions over destination footprints
    if dst_crs != src_crs:
        transformed_corners = projtools.reproject_points(
            (corners[..., 0].reshape(-1), corners[..., 1].reshape(-1)), in_crs=dst_crs, out_crs=src_crs
        )
        corners = np.column_stack(transformed_corners).reshape(-1, 4, 2)
    return centers[:, 0], centers[:, 1], corners


def _resolve_reprojection_operator(method: Any) -> Interpolator | Reducer:
    """Resolve resampling names to operators that can use local observation errors."""

    if isinstance(method, (Interpolator, Reducer)):
        return method
    name = method.name if isinstance(method, Resampling) else method

    # Named reducers for observation errors
    reducers: dict[RegularReductionMethod, type[Reducer]] = {
        "average": Mean,
        "sum": Sum,
        "min": Minimum,
        "max": Maximum,
    }
    operators: dict[str, type[Interpolator] | type[Reducer]] = {
        "nearest": Nearest,
        "bilinear": Linear,
    }
    operators.update(reducers.items())
    if name not in operators:
        raise ValueError(f"Observation errors for {name!r} require an explicit Interpolator or Reducer.")
    return operators[name]()


def _reproject_grid_operator(
    array: NDArrayNum,
    src_transform: rio.transform.Affine,
    src_crs: rio.crs.CRS,
    dst_transform: rio.transform.Affine,
    dst_crs: rio.crs.CRS,
    dst_shape: tuple[int, int],
    operator: Interpolator | Reducer,
    *,
    source_nodata: int | float | None,
    nodata_propagation: NodataPropagation,
    overlap_backend: OverlapBackend = "auto",
    coverage: GridCoverage | None = None,
    source_index_offset: tuple[int, int] = (0, 0),
    global_source_shape: tuple[int, int] | None = None,
    source_band_offset: int = 0,
) -> NDArrayNum:
    """
    Reproject from a regular raster using an Interpolator or Reducer for resampling.

    An Interpolator receives source cells around each transformed destination center.
    A Reducer without an explicit neighborhood uses the rectangular source window around a destination cell.
    Coverage selects centers, all touched cells, or fractional overlap with the destination cell. With GridNeighbours,
    it uses a fixed window around the cell containing each transformed destination center.

    The input can be single or multi-band (2D/3D array shape), and might be only a chunk
    (Dask/MP chunked logic sends here).

    Internal logic is:
    - _destination_grid_in_source_crs() locates the destination centers and footprints in the source CRS (to use fast
    regular grid interpolation or reduction).
    - Interpolators sample those centers with _interp_points_base()
    - Reducers with GridNeighbours sample those centers with _resample_at_points(). Other reducers use
    _grid_intersection_fractions_from_corners() to select source cells once for all bands.

    For fractional reductions, built-in methods can use _run_exactextract() or _reduce_overlap_batch() to evaluate many
    cells together; custom reductions and uncertainty use _grid_intersection_local_data() to describe each result's
    inputs.

    Return the reprojected array with the input's band layout.
    """

    # 1/ Derive destination cells coordinates in the source CRS
    source = np.asanyarray(array)
    source_bands = source[None, ...] if source.ndim == 2 else source
    global_shape = (
        (source_bands.shape[-2], source_bands.shape[-1]) if global_source_shape is None else global_source_shape
    )
    use_footprints = isinstance(operator, Reducer) and operator.default_neighborhood is None
    target_x, target_y, target_corners = _destination_grid_in_source_crs(
        dst_transform,
        dst_shape,
        dst_crs=dst_crs,
        src_crs=src_crs,
        include_corners=use_footprints,
    )
    source_indices: tuple[NDArrayNum, NDArrayNum] | None = None
    if isinstance(operator, Interpolator):
        # Preserve subpixel precision, including when coordinates are large relative to the pixel size
        source_rows, source_cols = _xy2ij(
            target_x,
            target_y,
            transform=src_transform,
            area_or_point=None,
            op=np.float64,
            shift_area_or_point=False,
        )
        # _xy2ij() measures from the upper-left corner; array values are located half a pixel inside each cell
        source_indices = (source_rows - 0.5, source_cols - 0.5)

    # Validate nodata handling and prepare outputs
    propagation = _validate_nodata_propagation(nodata_propagation)
    output_bands: list[NDArrayNum] = []
    handling: NodataHandling | None = None

    # 2/ Prepare source windows once for all bands when no explicit neighborhood is supplied

    reducer_overlap: GridIntersection | None = None
    reducer_polygons: NDArray[Any] | None = None
    exactextract_operation: str | None = None
    if use_footprints:
        assert isinstance(operator, Reducer) and target_corners is not None
        selected_coverage = "all_touched" if coverage is None else coverage
        if selected_coverage not in ("center", "all_touched", "fractional"):
            raise ValueError("coverage must be 'center', 'all_touched' or 'fractional'.")
        # Map projected corners to source pixel coordinates to bound the source window
        inverse = ~src_transform
        corner_columns = inverse.a * target_corners[..., 0] + inverse.b * target_corners[..., 1] + inverse.c
        corner_rows = inverse.d * target_corners[..., 0] + inverse.e * target_corners[..., 1] + inverse.f
        lower_columns, upper_columns = corner_columns.min(axis=1), corner_columns.max(axis=1)
        lower_rows, upper_rows = corner_rows.min(axis=1), corner_rows.max(axis=1)
        window_columns = np.stack((lower_columns, upper_columns, upper_columns, lower_columns), axis=1)
        window_rows = np.stack((lower_rows, lower_rows, upper_rows, upper_rows), axis=1)
        window_x = src_transform.a * window_columns + src_transform.b * window_rows + src_transform.c
        window_y = src_transform.d * window_columns + src_transform.e * window_rows + src_transform.f
        window_corners = np.stack((window_x, window_y), axis=-1)
        # Fractional coverage uses the projected cell itself to measure source pixel areas
        if selected_coverage == "fractional":
            window_corners = target_corners
        # Let ExactExtract calculate common statistics directly when its operation exactly matches the Reducer
        source_shape = (int(source_bands.shape[-2]), int(source_bands.shape[-1]))
        exactextract_operation = _exactextract_operation(operator)
        exactextract_compatible = (
            src_transform.b == 0 and src_transform.d == 0 and src_transform.a > 0 and src_transform.e < 0
        )
        # Uncertainty and nodata propagation need the individual covered cells, rather than a statistic alone
        # Aligned grids have a cheaper intersection calculation, so they also use the explicit fractions below
        direct_exactextract = (
            operator.error_structure is None
            and propagation != "propagate"
            and overlap_backend != "shapely"
            and exactextract_operation is not None
            and exactextract_compatible
            and find_spec("exactextract") is not None
            and selected_coverage == "fractional"
            and not _corners_are_grid_aligned(window_corners, src_transform)
        )
        if direct_exactextract:
            reducer_polygons = np.asarray(shapely.polygons(window_corners), dtype=object)
        else:
            # Calculate selected cells once because every band uses the same source and destination grids
            reducer_overlap = _grid_intersection_fractions_from_corners(
                window_corners,
                src_transform,
                source_shape,
                backend=overlap_backend,
            )
            if selected_coverage == "center":
                # Keep source cells whose centers fall inside the rectangular source window
                selected = []
                for target_index in range(reducer_overlap.geometry_count):
                    start, stop = reducer_overlap.offsets[target_index : target_index + 2]
                    rows = reducer_overlap.rows[start:stop]
                    columns = reducer_overlap.columns[start:stop]
                    selected.append(
                        (rows + 0.5 >= lower_rows[target_index])
                        & (rows + 0.5 < upper_rows[target_index])
                        & (columns + 0.5 >= lower_columns[target_index])
                        & (columns + 0.5 < upper_columns[target_index])
                    )
                use = np.concatenate(selected) if selected else np.empty(0, dtype=bool)
                counts = np.asarray([int(np.count_nonzero(mask)) for mask in selected], dtype=np.int64)
                offsets = np.r_[0, np.cumsum(counts)]
                reducer_overlap = GridIntersection(
                    offsets,
                    reducer_overlap.rows[use],
                    reducer_overlap.columns[use],
                    np.ones(int(offsets[-1])),
                )
            elif selected_coverage == "all_touched":
                reducer_overlap = replace(reducer_overlap, fractions=np.ones_like(reducer_overlap.fractions))

    # 3/ Calculate each band values

    for band_index, source_band in enumerate(source_bands):
        # Apply the requested source nodata value before running the GeoUtils calculation, as Rasterio does
        if source_nodata is not None:
            source_values = np.asanyarray(np.ma.getdata(source_band))
            source_mask = np.ma.getmaskarray(source_band)
            if np.isnan(source_nodata):
                source_mask = source_mask | ~np.isfinite(source_values)
            else:
                source_mask = source_mask | (source_values == source_nodata)
            source_band = np.ma.masked_array(source_values, mask=source_mask)
        source_id_offset = (band_index + source_band_offset) * int(np.prod(global_shape))

        if isinstance(operator, Interpolator):
            # Interpolate the transformed destination centers with the same method as interp_at_points()
            band_output = _interp_points_base(
                source_band,
                transform=src_transform,
                points=(target_x, target_y),
                area_or_point=None,
                method=operator,
                shift_area_or_point=False,
                nodata_propagation=propagation,
                array_indices=source_indices,
                source_index_offset=source_index_offset,
                source_shape=global_shape,
                source_band=band_index + source_band_offset + 1,
            )
        else:
            # Use the fixed source window or the window derived from each destination cell
            handling = "ignore" if propagation == "gdal" else propagation
            if isinstance(operator.default_neighborhood, GridNeighbours):
                # Reuse point sampling so offsets, nodata rules and band IDs match resample_at_points()
                neighborhood = operator.default_neighborhood
                window_size = 2 * max(neighborhood.overlap) + 1
                if neighborhood.coverage != "center" and neighborhood.window_shape is None:
                    raise ValueError("Area coverage requires a square or circular GridNeighbours window.")
                band_output = _resample_at_points(
                    source_band,
                    src_transform,
                    (target_x, target_y),
                    operator,
                    area_or_point=None,
                    shift_area_or_point=False,
                    nodata_propagation=propagation,
                    dist_nodata_spread=None,
                    neighborhood=neighborhood,
                    fractional_window=window_size if neighborhood.coverage != "center" else None,
                    fractional_shape=neighborhood.window_shape,
                    band=band_index + source_band_offset + 1,
                    source_index_offset=source_index_offset,
                    source_shape=global_shape,
                )
            elif reducer_polygons is not None:
                # ExactExtract calculates common statistics without returning every overlap fraction to Python
                assert exactextract_operation is not None
                band_output = _run_exactextract(
                    reducer_polygons,
                    src_transform,
                    source_band.shape,
                    values=np.asanyarray(source_band),
                    operation=exactextract_operation,
                )
            elif type(operator) in (Mean, Sum, Minimum, Maximum) and operator.error_structure is None:
                # These built-ins can reduce all groups at once using the already calculated overlap fractions
                assert reducer_overlap is not None
                band_output = _reduce_overlap_batch(
                    source_band,
                    reducer_overlap,
                    cast(Mean | Sum | Minimum | Maximum, operator),
                    nodata_propagation=handling,
                )
            else:
                # Custom reducers and uncertainty need each destination cell's values, area fractions and source IDs
                assert reducer_overlap is not None
                targets = np.column_stack((target_x, target_y))
                local_inputs = _grid_intersection_local_data(
                    source_band,
                    src_transform,
                    reducer_overlap,
                    source_id_offset=source_id_offset,
                    source_index_offset=source_index_offset,
                    source_shape=global_shape,
                    source_nodata=source_nodata,
                    targets=targets,
                )
                if selected_coverage != "fractional":
                    local_inputs = [replace(local, support_weights=None) for local in local_inputs]
                band_output = _evaluate_operator_batch(operator, local_inputs, nodata_propagation=handling)
        output_bands.append(np.asarray(band_output).reshape(dst_shape))

    # 4/ Restore the original band layout

    output = output_bands[0] if source.ndim == 2 else np.stack(output_bands)
    return output


# 2.3/ CHUNKED LOGIC (for both Dask and Multiprocessing)

# At the date of April 2024: not supported by Rioxarray
# Part of the code was inspired by https://github.com/opendatacube/odc-geo/pull/88, modified to be concise,
# stand-alone and rely only on Rasterio/Shapely/Pyproj


def _combined_blocks_shape_transform(
    sub_block_ids: list[dict[str, int]], src_geogrid: GeoGrid
) -> tuple[dict[str, Any], list[dict[str, int]]]:
    """Derive combined shape and transform from a subset of several blocks (for source input during reprojection)."""

    # Get combined shape by taking min of X/Y starting indices, max of X/Y ending indices
    all_xs, all_ys, all_xe, all_ye = ([b[s] for b in sub_block_ids] for s in ["xs", "ys", "xe", "ye"])
    minmaxs = {"min_xs": np.min(all_xs), "max_xe": np.max(all_xe), "min_ys": np.min(all_ys), "max_ye": np.max(all_ye)}
    combined_shape = (minmaxs["max_ye"] - minmaxs["min_ys"], minmaxs["max_xe"] - minmaxs["min_xs"])

    # Shift source transform with start indexes to get the one for combined block location
    combined_transform = src_geogrid.translate(xoff=minmaxs["min_xs"], yoff=-minmaxs["min_ys"]).transform

    # Compute relative block indexes that will be needed to reconstruct a square array in the delayed function,
    # by subtracting the minimum starting indices in X/Y
    relative_block_indexes = [
        {"r" + s1 + s2: b[s1 + s2] - minmaxs["min_" + s1 + "s"] for s1 in ["x", "y"] for s2 in ["s", "e"]}
        for b in sub_block_ids
    ]

    combined_meta = {"src_shape": combined_shape, "src_transform": tuple(combined_transform)}

    return combined_meta, relative_block_indexes


def _expand_source_block_indices(
    block_indices: list[int],
    src_block_ids: list[dict[str, Any]],
    src_numblocks: tuple[int, int],
    margin: tuple[int, int],
) -> list[int]:
    """
    Expand source blocks by a fixed number of neighboring chunks in each direction.

    Rasterio's reprojection can depend on pixels just outside the destination/source overlap, especially at chunk
    boundaries and with non-nearest kernels. Methods using fixed GridNeighbours offsets read enough chunks to include
    their exact row and column overlap without materializing the entire source raster.
    """

    ny, nx = src_numblocks
    by_location = {tuple(block_id["chunk-location"]): i for i, block_id in enumerate(src_block_ids)}
    expanded = set(block_indices)

    for index in block_indices:
        iy, ix = src_block_ids[index]["chunk-location"]
        for dy in range(-margin[0], margin[0] + 1):
            for dx in range(-margin[1], margin[1] + 1):
                jy = int(iy) + dy
                jx = int(ix) + dx
                if 0 <= jy < ny and 0 <= jx < nx:
                    expanded.add(by_location[(jy, jx)])

    return sorted(expanded)


def _build_geotiling_and_meta(
    src_count: int,
    src_shape: tuple[int, int],
    src_transform: rio.transform.Affine,
    src_crs: CRS,
    dst_shape: tuple[int, int],
    dst_transform: rio.transform.Affine,
    dst_crs: CRS,
    src_chunks: tuple[tuple[int, ...], tuple[int, ...]],
    dst_chunksizes: tuple[int, int],
    source_pixel_overlap: tuple[int, int] = (0, 0),
) -> tuple[
    ChunkedGeoGrid,
    ChunkedGeoGrid,
    tuple[tuple[int, ...], tuple[int, ...]],
    list[list[int]],
    list[dict[str, int]],
    list[tuple[dict[str, Any], list[dict[str, int]]]],
    list[GeoGrid],
]:
    """
    Constructs georeferenced tiling information and reprojection metadata for both source and destination grids,
    used to support block-wise reprojection operations (e.g. with multiprocessing or dask).

    This function performs the following:

    1. Constructs `GeoGrid` and `ChunkedGeoGrid` objects for source and destination rasters,
       based on provided shape, transform, CRS, and chunk sizes.
    2. Computes spatial footprints for each chunk in both grids, and determines which
       source chunks intersect each destination chunk (with a buffer to ensure overlap).
    3. For each destination chunk, calculates metadata required for reprojection, including:
       - The combined shape and transform of all intersecting source chunks.
       - The specific shape and transform of the destination block.

    :return: A tuple containing:
        - Source `ChunkedGeoGrid`
        - Destination `ChunkedGeoGrid`
        - Destination chunks
        - Mapping from destination to intersecting source block indices
        - Array of source block locations
        - List of metadata dictionaries per destination block
        - List of destination `GeoGrid` blocks
    """
    # 1/ Define source and destination chunked georeferenced grid through simple classes storing CRS/transform/shape,
    # which allow to consistently derive shape/transform for each block and their CRS-projected footprints

    # Define GeoGrids for source/destination array
    src_geogrid = GeoGrid(transform=src_transform, shape=src_shape, crs=src_crs)
    dst_geogrid = GeoGrid(transform=dst_transform, shape=dst_shape, crs=dst_crs)

    # Create tilings
    src_geotiling = ChunkedGeoGrid(grid=src_geogrid, chunks=src_chunks)
    dst_chunks = normalize_chunks(chunks=dst_chunksizes, shape=dst_shape)
    dst_geotiling = ChunkedGeoGrid(grid=dst_geogrid, chunks=dst_chunks)

    # 2/ Get bounds of tiles in CRS of destination array, with a buffer of 2 pixels for destination ones to ensure
    # overlap, then map indexes of source blocks that intersect a given destination block
    src_boxes = [box(*gg.bounds_projected(crs=dst_crs)) for gg in src_geotiling.get_blocks_as_geogrids()]
    dst_boxes = [
        box(*gg.bounds_projected(crs=dst_crs)).buffer(2 * max(dst_geogrid.res))
        for gg in dst_geotiling.get_blocks_as_geogrids()
    ]
    # Faster to use spatial index over source boxes
    tree = STRtree(src_boxes)
    # For Shapely 2.0: STRtree.query(..., predicate="intersects") is fastest, for earlier versions we filter manually

    # Quick feature check
    try:
        _ = tree.query(dst_boxes[0], predicate="intersects") if dst_boxes else []
        has_predicate = True
    except TypeError:
        has_predicate = False

    # Build mapping: for each destination box, list intersecting source indices
    dest2source: list[list[int]] = []
    if has_predicate:
        # Shapely 2: Query returns indices directly (int array)
        for dst in dst_boxes:
            idx = tree.query(dst, predicate="intersects")
            dest2source.append([int(i) for i in np.asarray(idx).ravel()])
    else:
        # Shapely 1.8: Query returns geometries, so we convert to indices via id() map + filter intersects
        id_to_idx = {id(g): i for i, g in enumerate(src_boxes)}
        for dst in dst_boxes:
            cand_geoms = tree.query(dst)
            matches = [id_to_idx[id(g)] for g in cand_geoms if dst.intersects(g)]
            dest2source.append(matches)

    # 3/ To reconstruct a square source array during chunked reprojection, we need to derive the combined shape and
    # transform of each tuples of source blocks
    src_block_ids = src_geotiling.get_block_locations()
    # Rasterio reads one neighboring chunk; GridNeighbours offsets may require several small source chunks
    chunk_margin = (
        max(1, int(np.ceil(source_pixel_overlap[0] / min(src_chunks[0])))),
        max(1, int(np.ceil(source_pixel_overlap[1] / min(src_chunks[1])))),
    )
    dest2source = [
        _expand_source_block_indices(
            sbid,
            src_block_ids=src_block_ids,
            src_numblocks=src_geotiling.num_chunks,
            margin=chunk_margin,
        )
        for sbid in dest2source
    ]
    meta_params = [
        (
            _combined_blocks_shape_transform(sub_block_ids=[src_block_ids[i] for i in sbid], src_geogrid=src_geogrid)
            if len(sbid) > 0
            else ({}, [])
        )
        for sbid in dest2source
    ]

    # Append dst shape/transform to metadata
    dst_block_geogrids = dst_geotiling.get_blocks_as_geogrids()
    for i, (c, _) in enumerate(meta_params):
        c.update(
            {
                "dst_shape": dst_block_geogrids[i].shape,
                "dst_transform": tuple(dst_block_geogrids[i].transform),
                "dst_count": src_count,
                "global_source_shape": src_shape,
                "global_source_transform": src_transform,
            }
        )

    return src_geotiling, dst_geotiling, dst_chunks, dest2source, src_block_ids, meta_params, dst_block_geogrids


def _reproject_per_block(
    *src_arrs: tuple[NDArrayNum],
    block_ids: list[dict[str, int]],
    combined_meta: dict[str, Any],
    source_band_offset: int = 0,
    **kwargs: Any,
) -> NDArrayNum:
    """
    Reprojection per destination block (also rebuilds a square array combined from intersecting source blocks).
    """

    # A single-band Xarray block still has a leading band dimension, so include it in the delayed array shape
    is_multiband = src_arrs[0].ndim == 3 if src_arrs else combined_meta["dst_count"] >= 2

    # If no source chunk intersects, we return a chunk of destination nodata values
    if len(src_arrs) == 0:
        # We can use float32 to return NaN, will be cast to other floating type later if that's not source array dtype
        dst_shape = (
            (combined_meta["dst_count"], *combined_meta["dst_shape"]) if is_multiband else combined_meta["dst_shape"]
        )
        dst_arr = np.zeros(dst_shape, dtype=np.dtype("float32"))
        dst_arr[:] = np.nan
        return dst_arr

    # First, we build an empty array with the combined shape, only with nodata values
    shape = (src_arrs[0].shape[0], *combined_meta["src_shape"]) if is_multiband else combined_meta["src_shape"]

    comb_src_arr = np.full(shape, kwargs["src_nodata"], dtype=src_arrs[0].dtype)
    if np.ma.isMaskedArray(src_arrs[0]):
        comb_src_arr = np.ma.masked_array(data=comb_src_arr)

    # Then fill it with the source chunks values
    for arr, bid in zip(src_arrs, block_ids):
        comb_src_arr[..., bid["rys"] : bid["rye"], bid["rxs"] : bid["rxe"]] = arr

    # Build the combined transforms before choosing Rasterio or the GeoUtils Interpolator and Reducer path
    src_transform = rio.transform.Affine(*combined_meta["src_transform"])
    dst_transform = rio.transform.Affine(*combined_meta["dst_transform"])

    if isinstance(kwargs["resampling"], (Interpolator, Reducer)):
        operator = kwargs["resampling"]
        propagation = _validate_nodata_propagation(kwargs.get("nodata_propagation", "gdal"))
        # Error-model IDs refer to the full raster even when workers read a smaller source rectangle
        source_row, source_col = _xy2ij(
            src_transform.c,
            src_transform.f,
            transform=combined_meta["global_source_transform"],
            area_or_point=None,
            shift_area_or_point=False,
            op=np.float64,
        )
        operator_result = _reproject_grid_operator(
            comb_src_arr,
            src_transform=src_transform,
            src_crs=kwargs["src_crs"],
            dst_transform=dst_transform,
            dst_crs=kwargs["dst_crs"],
            dst_shape=combined_meta["dst_shape"],
            operator=operator,
            source_nodata=kwargs.get("src_nodata"),
            nodata_propagation=propagation,
            overlap_backend=kwargs.get("overlap_backend", "auto"),
            coverage=kwargs.get("coverage"),
            source_index_offset=(int(round(np.asarray(source_row).item())), int(round(np.asarray(source_col).item()))),
            global_source_shape=combined_meta["global_source_shape"],
            source_band_offset=source_band_offset,
        )

        # Use a mask for nodata results when an integer output array cannot represent NaN
        output_dtype = np.dtype(kwargs.get("dtype", comb_src_arr.dtype))
        invalid = ~np.isfinite(operator_result)
        if np.any(invalid) and not np.issubdtype(output_dtype, np.floating):
            fill_value = kwargs.get("dst_nodata")
            if fill_value is None:
                fill_value = _default_nodata(output_dtype)
            values = np.where(invalid, fill_value, operator_result).astype(output_dtype)
            return np.ma.masked_array(values, mask=invalid, fill_value=fill_value)
        return np.asarray(operator_result, dtype=output_dtype)

    # Call Rasterio for its built-in Resampling modes
    # Force the number of threads to 1 to avoid Dask/Rasterio conflicting on multi-threading
    kwargs.update(
        {
            "dst_shape": combined_meta["dst_shape"],
            "src_transform": src_transform,
            "dst_transform": dst_transform,
            "num_threads": 1,
        }
    )
    # Define dtype if undefined
    if "dtype" not in kwargs:
        kwargs.update({"dtype": comb_src_arr.dtype})

    dst_arr = _rio_reproject(src_arr=comb_src_arr, reproj_kwargs=kwargs)  # type: ignore

    return dst_arr


@delayed
def _delayed_reproject_per_block(
    *src_arrs: tuple[NDArrayNum], block_ids: list[dict[str, int]], combined_meta: dict[str, Any], **kwargs: Any
) -> NDArrayNum:
    """
    Delayed reprojection per destination block (also rebuilds a square array combined from intersecting source blocks).
    """
    return _reproject_per_block(*src_arrs, block_ids=block_ids, combined_meta=combined_meta, **kwargs)


def _dask_reproject(
    darr: da.Array,
    src_transform: rio.transform.Affine,
    src_crs: rio.crs.CRS,
    dst_transform: rio.transform.Affine,
    dst_shape: tuple[int, int],
    dst_crs: rio.crs.CRS,
    resampling: rio.enums.Resampling | Interpolator | Reducer,
    src_nodata: int | float | None = None,
    dst_nodata: int | float | None = None,
    dst_chunksizes: tuple[int, int] | None = None,
    source_pixel_overlap: tuple[int, int] = (0, 0),
    **kwargs: Any,
) -> da.Array:
    """
    Reproject georeferenced raster on out-of-memory chunks.

    Each chunk of the destination array is mapped to one or several intersecting chunks of the source array, and
    reprojection is performed using rio.warp.reproject for each mapping.

    Part of the code is inspired by https://github.com/opendatacube/odc-geo/pull/88.

    :param darr: Input dask array for source raster.
    :param src_transform: Geotransform of source raster.
    :param src_crs: Coordinate reference system of source raster.
    :param dst_transform: Geotransform of destination raster.
    :param dst_shape: Shape of destination raster.
    :param dst_crs: Coordinate reference system of destination raster.
    :param resampling: Resampling method.
    :param src_nodata: Nodata value of source raster.
    :param dst_nodata: Nodata value of destination raster.
    :param dst_chunksizes: Chunksizes for destination raster.
    :param kwargs: Other arguments to pass to rio.warp.reproject().

    :return: Dask array of reprojected raster.
    """

    # To raise appropriate error on missing optional dependency
    import_optional("dask")

    # Define the chunking
    # For source, we can use the .chunks attribute
    src_chunks = darr.chunks[-2:]  # In case input is multi-band

    if dst_chunksizes is None:
        dst_chunksizes = (darr.chunksize[-2], darr.chunksize[-1])  # In case input is multi-band

    # Prepare geotiling and reprojection metadata for source and destination grids
    src_geotiling, dst_geotiling, dst_chunks, dest2source, src_block_ids, meta_params, dst_block_geogrids = (
        _build_geotiling_and_meta(
            src_count=darr.shape[0] if darr.ndim == 3 else 1,
            src_shape=darr.shape[-2:],  # In case input is multi-band
            src_transform=src_transform,
            src_crs=src_crs,
            dst_shape=dst_shape,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_chunks=src_chunks,
            dst_chunksizes=dst_chunksizes,
            source_pixel_overlap=source_pixel_overlap,
        )
    )

    # We call a delayed function that uses rio.warp to reproject the combined source block(s) to each destination block

    # Add fixed arguments to keywords
    kwargs.update(
        {
            "src_nodata": src_nodata,
            "dst_nodata": dst_nodata,
            "resampling": resampling,
            "src_crs": src_crs,
            "dst_crs": dst_crs,
        }
    )

    # Create a delayed object for each block, and flatten the blocks into a 1d shape
    blocks_delayed = darr.to_delayed()

    # Spatial block grid shape (from spatial chunks)
    is_multiband = darr.ndim == 3
    ny_src = len(src_chunks[0])
    nx_src = len(src_chunks[1])
    src_yi, src_xi = np.unravel_index(np.arange(ny_src * nx_src), shape=(ny_src, nx_src))
    # Normalize band groups:
    # - 2D: one pseudo group (bb=None, nb=0)
    # - 3D: real band blocks with their sizes
    band_groups: list[tuple[int | None, int]] = (
        [(None, 0)] if not is_multiband else [(bb, int(sz)) for bb, sz in enumerate(darr.chunks[0])]
    )
    # Output data type
    out_dtype = np.dtype(kwargs.get("dtype", darr.dtype))

    # Helper function to support both 2D and 3D cases
    def _dst_block_as_da(i: int) -> da.Array:
        """Build destination block as a Dask array (2D or 3D)."""
        shp2 = dst_block_geogrids[i].shape  # (ydst, xdst)

        # Spatial source coords for this destination tile
        coords = [(src_yi[j], src_xi[j]) for j in dest2source[i]]

        def _src_chunks_for_group(bb: int | None) -> list[Any]:
            # Accounting for the fact that blocks_delayed is either (ny,nx) or (nb,ny,nx)
            if bb is None:
                return [blocks_delayed[y, x] for (y, x) in coords]
            return [blocks_delayed[bb, y, x] for (y, x) in coords]

        def _one_group(bb: int | None, nb: int) -> da.Array:
            r = _delayed_reproject_per_block(
                *_src_chunks_for_group(bb),
                block_ids=meta_params[i][1],
                combined_meta=meta_params[i][0],
                source_band_offset=0 if bb is None else sum(darr.chunks[0][:bb]),
                **kwargs,
            )
            shape = shp2 if bb is None else (nb, *shp2)
            # We define the expected output shape and dtype to simplify things for Dask
            return da.from_delayed(r, shape=shape, dtype=out_dtype)

        # Build per-group outputs then concatenate along band axis if needed
        groups = [_one_group(bb, nb) for (bb, nb) in band_groups]
        return groups[0] if len(groups) == 1 else da.concatenate(groups, axis=0)

    # Run the delayed reprojection, looping for each destination block-band (2D block and 1D band-chunk)
    list_reproj_da = [_dst_block_as_da(i) for i in range(len(dest2source))]

    # Array comes out as flat blocks x chunksize0 (varying) x chunksize1 (varying), so we can't reshape directly
    # We need to unravel the flattened blocks indices to align X/Y, then concatenate all columns, then rows
    ny_dst, nx_dst = len(dst_chunks[0]), len(dst_chunks[1])
    iy, ix = np.unravel_index(np.arange(len(dest2source)), shape=(ny_dst, nx_dst))
    ax_x = 1 if darr.ndim == 2 else 2  # Adjust axes depending on if raster is single-band or multi-band
    ax_y = 0 if darr.ndim == 2 else 1
    rows = [
        da.concatenate([list_reproj_da[k] for k in range(len(list_reproj_da)) if iy[k] == r], axis=ax_x)
        for r in range(ny_dst)
    ]
    concat_all = da.concatenate(rows, axis=ax_y)
    return concat_all


def _wrapper_multiproc_reproject_per_block(
    rst: Raster,
    src_block_ids: list[dict[str, int]],
    dst_block_id: dict[str, int],
    idx_d2s: list[int],
    block_ids: list[dict[str, int]],
    combined_meta: dict[str, Any],
    **kwargs: Any,
) -> tuple[NDArrayNum, tuple[int, int, int, int]]:
    """Wrapper to use Delayed reprojection per destination block
    (also rebuilds a square array combined from intersecting source blocks)."""

    # Get source array block for each destination block
    s = src_block_ids
    src_arrs = (rst.icrop(bbox=(s[idx]["xs"], s[idx]["ys"], s[idx]["xe"], s[idx]["ye"])).data for idx in idx_d2s)

    # Call reproject per block
    dst_block_arr = _reproject_per_block(*src_arrs, block_ids=block_ids, combined_meta=combined_meta, **kwargs)

    # Store logical masks as integers so the writer can fill missing cells with nodata rather than True
    if dst_block_arr.dtype == np.bool_:
        dst_block_arr = dst_block_arr.astype("uint8")

    return dst_block_arr, (dst_block_id["ys"], dst_block_id["ye"], dst_block_id["xs"], dst_block_id["xe"])


def _multiproc_reproject(
    rst: Raster,
    mp_config: MultiprocConfig,
    src_crs: rio.CRS,
    src_nodata: int | float | None,
    dst_shape: tuple[int, int],
    dst_transform: rio.Affine,
    dst_crs: rio.CRS,
    dst_nodata: int | float | None,
    dtype: DTypeLike,
    resampling: Resampling | Interpolator | Reducer,
    source_pixel_overlap: tuple[int, int] = (0, 0),
    **kwargs: Any,
) -> None:
    """
    Reproject georeferenced raster on out-of-memory chunks with multiprocessing.
    See Raster.reproject() for details.
    """

    # Prepare geotiling and reprojection metadata for source and destination grids
    src_chunks = normalize_chunks(chunks=mp_config.chunks, shape=rst.shape)
    src_geotiling, dst_geotiling, dst_chunks, dest2source, src_block_ids, meta_params, dst_block_geogrids = (
        _build_geotiling_and_meta(
            src_count=rst.count,
            src_shape=rst.shape,
            src_transform=rst.transform,
            src_crs=rst.crs,
            dst_shape=dst_shape,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_chunks=src_chunks,
            dst_chunksizes=_split_chunk_size(mp_config.chunks),
            source_pixel_overlap=source_pixel_overlap,
        )
    )

    # 4/ Call a delayed function that uses rio.warp to reproject the combined source block(s) to each destination block
    kwargs.update(
        {
            "src_nodata": src_nodata,
            "dst_nodata": dst_nodata,
            "resampling": resampling,
            "src_crs": src_crs,
            "dst_crs": dst_crs,
        }
    )
    # Get location of destination blocks to write file
    dst_block_ids = np.array(dst_geotiling.get_block_locations())

    # Create tasks for multiprocessing
    tasks = []
    for i in range(len(dest2source)):
        tasks.append(
            mp_config.cluster.submit(
                _wrapper_multiproc_reproject_per_block,
                rst,
                src_block_ids,
                dst_block_ids[i],
                dest2source[i],
                meta_params[i][1],
                meta_params[i][0],
                **kwargs,
            )
        )

    # Retrieve metadata for saving file
    file_metadata = {
        "width": dst_shape[1],
        "height": dst_shape[0],
        "count": rst.count,
        "crs": dst_crs,
        "transform": dst_transform,
        "dtype": "uint8" if np.dtype(dtype) == np.bool_ else dtype,
        "nodata": dst_nodata,
    }

    # Create a new raster file to save the processed results
    _write_multiproc_result(tasks, mp_config, file_metadata)


def _reproject(
    source_raster: RasterType,
    ref: RasterLike,
    crs: CRS | str | int | None = None,
    res: float | tuple[float, float] | None = None,
    grid_size: tuple[int, int] | None = None,
    bounds: tuple[float, float, float, float] | rio.coords.BoundingBox | None = None,
    nodata: int | float | None = None,
    dtype: DTypeLike | None = None,
    resampling: Resampling | str | Interpolator | Reducer = None,
    force_source_nodata: int | float | None = None,
    silent: bool = False,
    n_threads: int = 0,
    memory_limit: int = 64,
    mp_config: MultiprocConfig | None = None,
    nodata_propagation: NodataPropagation = "gdal",
    overlap_backend: OverlapBackend = "auto",
    window: int | None = None,
    window_shape: Literal["square", "circular"] | None = None,
    coverage: GridCoverage | None = None,
) -> Any:
    """
    Reproject raster. See Raster.reproject() for details.
    """

    # If resampling method undefined, default to the global system config
    if resampling is None:
        resampling = config["reprojection_method"]

    # 1/ Check and normalize match-grid inputs
    _check_crs(source_raster.crs)
    dst_shape, dst_transform, dst_crs = _check_match_grid(
        src=source_raster, ref=ref, res=res, shape=grid_size, bounds=bounds, crs=crs, coords=None
    )

    # 2/ Check user input for nodata and dtype
    dtype, src_nodata, nodata = _check_reproj_nodata_dtype(
        source_raster=source_raster,
        nodata=nodata,
        dtype=dtype,
        force_source_nodata=force_source_nodata,
    )

    # 3/ Store georeferencing parameters and convert named methods to Rasterio's Resampling values
    is_operator = isinstance(resampling, (Interpolator, Reducer))
    if isinstance(resampling, Interpolator) and (
        window is not None or window_shape is not None or coverage is not None
    ):
        raise ValueError("Source window options require a Reducer.")
    if isinstance(resampling, Interpolator):
        regular_method = _regular_interpolation_method(resampling)
        if regular_method is not None:
            _check_regular_grid_neighbours(resampling, regular_method, warn=True)
        else:
            neighbours = _resolve_grid_neighbours_for_interpolator(resampling, source_raster.transform)
            if resampling.default_neighborhood is None:
                raster_operator = copy(resampling)
                raster_operator.default_neighborhood = neighbours
                resampling = raster_operator
    elif isinstance(resampling, Reducer):
        if isinstance(resampling.default_neighborhood, PointNeighbours):
            raise ValueError("PointNeighbours applies to point sources; use GridNeighbours for raster cells.")
        neighborhood = _configure_grid_neighbours(
            resampling.default_neighborhood,
            size=window,
            shape=window_shape,
            coverage=coverage,
            default_size=3 if window_shape is not None else None,
        )
        if neighborhood is not None and neighborhood is not resampling.default_neighborhood:
            configured = copy(resampling)
            configured.default_neighborhood = neighborhood
            resampling = configured
    elif window is not None or window_shape is not None or coverage is not None:
        raise ValueError("Source window options require a Reducer.")
    if is_operator:
        resolved_resampling = Resampling.nearest
    else:
        resolved_resampling = (
            resampling if isinstance(resampling, Resampling) else _resampling_method_from_str(cast(str, resampling))
        )
    reproj_kwargs = {
        "src_transform": source_raster.transform,
        "dst_transform": dst_transform,
        "src_crs": source_raster.crs,
        "dst_crs": dst_crs,
        "resampling": resolved_resampling,
        "src_nodata": src_nodata,
        "dst_nodata": nodata,
        "dtype": dtype,
        "dst_shape": dst_shape,
    }

    # 4/ Check if reprojection is needed, otherwise return source raster with warning
    if not is_operator and _is_reproj_needed(src_shape=source_raster.shape, reproj_kwargs=reproj_kwargs):
        if (nodata == src_nodata) or (nodata is None):
            if not silent:
                warnings.warn("Output projection, bounds and grid size are identical -> returning self (not a copy!)")
            return True, None, None, None, None

        elif nodata is not None:
            if not silent:
                warnings.warn(
                    "Only nodata is different, consider using the 'set_nodata()' method instead'\
                ' -> returning self (not a copy!)"
                )
            return True, None, None, None, None

    # 5/ Perform reprojection
    reproj_kwargs.update({"num_threads": n_threads, "warp_mem_limit": memory_limit})
    # Cannot use Multiprocessing backend and Dask backend simultaneously
    mp_backend = mp_config is not None
    # The check below can only run on Xarray
    dask_backend = da is not None and source_raster._chunks is not None

    if mp_backend and dask_backend:
        raise ValueError(
            "Cannot use Multiprocessing and Dask simultaneously. To use Dask, remove mp_config parameter "
            "from reproject(). To use Multiprocessing, open the file without chunks."
        )

    # Apply an Interpolator or Reducer in GeoUtils because Rasterio only accepts its own Resampling values
    if is_operator and not mp_backend and not dask_backend:
        propagation = _validate_nodata_propagation(nodata_propagation)
        dst_arr = _reproject_grid_operator(
            source_raster.data,
            src_transform=source_raster.transform,
            src_crs=source_raster.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_shape=dst_shape,
            operator=cast(Interpolator | Reducer, resampling),
            source_nodata=src_nodata,
            nodata_propagation=propagation,
            overlap_backend=overlap_backend,
            coverage=coverage,
        )
        # Use a mask for missing results when the requested integer data type cannot represent NaN
        output_dtype = np.dtype(dtype)
        invalid = ~np.isfinite(dst_arr)
        if np.any(invalid) and not np.issubdtype(output_dtype, np.floating):
            fill_value = nodata if nodata is not None else _default_nodata(output_dtype)
            cast_values = np.where(invalid, fill_value, dst_arr).astype(output_dtype)
            dst_arr = np.ma.masked_array(cast_values, mask=invalid, fill_value=fill_value)
        else:
            dst_arr = np.asarray(dst_arr, dtype=output_dtype)
        result = False, dst_arr, dst_transform, dst_crs, nodata
        return result

    # Chunked Interpolators and Reducers use the existing block mapping and read any extra cells required at the edges
    if is_operator:
        source_pixel_overlap = (0, 0)
        operator = cast(Interpolator | Reducer, resampling)
        if isinstance(operator.default_neighborhood, GridNeighbours):
            source_pixel_overlap = operator.default_neighborhood.overlap
        reproj_kwargs.update(
            {
                "resampling": operator,
                "nodata_propagation": _validate_nodata_propagation(nodata_propagation),
                "overlap_backend": overlap_backend,
                "coverage": coverage,
                "source_pixel_overlap": source_pixel_overlap,
            }
        )

    # If using Multiprocessing backend, process and return None (files written on disk)
    if mp_config is not None:
        _multiproc_reproject(source_raster, mp_config=mp_config, **reproj_kwargs)  # type: ignore
        return False, None, None, None, None

    # If using Dask backend, process and return Dask array
    if da is not None and isinstance(source_raster.data, da.Array):
        dst_arr = _dask_reproject(darr=source_raster.data, **reproj_kwargs)

    # If using direct reprojection, process and return NumPy array
    else:
        dst_arr = _rio_reproject(src_arr=source_raster.data, reproj_kwargs=reproj_kwargs)

    result = False, dst_arr, reproj_kwargs["dst_transform"], reproj_kwargs["dst_crs"], reproj_kwargs["dst_nodata"]
    return result


#########
# 3/ CROP
#########


def _crop_window(
    source_raster: RasterType,
    bbox: RasterLike | VectorLike | tuple[float, float, float, float],
    distance_unit: Literal["georeferenced", "pixel"] = "georeferenced",
) -> tuple[rio.windows.Window, affine.Affine]:
    """Return the aligned source window and transform selected by a bounding box."""

    # Check input, raise appropriate errors and warnings
    bbox = _check_match_bbox(source_raster, bbox)

    assert distance_unit in ["georeferenced", "pixel"], "distance_unit must be 'georeferenced' or 'pixel'"

    # If using georeferenced unit, use bbox directly
    if distance_unit == "georeferenced":
        xmin, ymin, xmax, ymax = bbox
    # Else, convert the pixel window's corners to coordinates
    else:
        colmin, rowmin, colmax, rowmax = bbox
        (xmin, xmax), (ymax, ymin) = _ij2xy(
            np.asarray([rowmin, rowmax]),
            np.asarray([colmin, colmax]),
            transform=source_raster.transform,
            area_or_point=None,
            shift_area_or_point=False,
            force_offset="ul",
        )

    # Finding the intersection of requested bounds and original bounds, cropped to image shape
    ref_win = rio.windows.from_bounds(xmin, ymin, xmax, ymax, transform=source_raster.transform)
    self_win = rio.windows.from_bounds(*source_raster.bbox, transform=source_raster.transform).crop(
        *source_raster.shape
    )
    final_window = ref_win.intersection(self_win).round_lengths().round_offsets()

    # Update bounds and transform accordingly
    new_xmin, new_ymin, new_xmax, new_ymax = rio.windows.bounds(final_window, transform=source_raster.transform)
    tfm = rio.transform.from_origin(new_xmin, new_ymax, *source_raster.res)
    return final_window, tfm


def _crop(
    source_raster: RasterType,
    bbox: RasterLike | VectorLike | tuple[float, float, float, float],
    distance_unit: Literal["georeferenced", "pixel"] = "georeferenced",
) -> tuple[NDArrayNum, affine.Affine]:
    """Read or select the raster window requested by crop() or icrop()."""

    final_window, tfm = _crop_window(source_raster=source_raster, bbox=bbox, distance_unit=distance_unit)

    if source_raster._is_xr:
        (rowmin, rowmax), (colmin, colmax) = final_window.toranges()
        assert source_raster._obj is not None
        crop_img = source_raster._obj.isel(y=slice(rowmin, rowmax), x=slice(colmin, colmax))

    elif source_raster.is_loaded:
        # In case data is loaded on disk, can extract directly from np array
        (rowmin, rowmax), (colmin, colmax) = final_window.toranges()
        crop_img = source_raster.data[..., rowmin:rowmax, colmin:colmax]

    else:
        assert source_raster._disk_shape is not None  # This should not be the case, sanity check to make mypy happy

        # If data was not loaded, and self's transform was updated (e.g. due to downsampling) need to
        # get the Window corresponding to on disk data
        new_xmin, new_ymin, new_xmax, new_ymax = rio.windows.bounds(final_window, transform=source_raster.transform)
        ref_win_disk = rio.windows.from_bounds(
            new_xmin, new_ymin, new_xmax, new_ymax, transform=source_raster._disk_transform
        )
        self_win_disk = rio.windows.from_bounds(*source_raster.bbox, transform=source_raster._disk_transform).crop(
            *source_raster._disk_shape[1:]
        )
        final_window_disk = ref_win_disk.intersection(self_win_disk).round_lengths().round_offsets()

        # Round up to downsampling size, to match __init__
        final_window_disk = rio.windows.round_window_to_full_blocks(
            final_window_disk, ((source_raster._downsample, source_raster._downsample),)
        )

        with ExitStack() as stack:
            source = stack.enter_context(rio.open(source_raster.name))
            if source_raster._downsample > 1:
                raster = stack.enter_context(_open_downsampled_raster(source, source_raster._downsample))
                source_window = source_raster._out_window or rio.windows.Window(
                    0, 0, source_raster.width, source_raster.height
                )
                read_window = rio.windows.Window(
                    source_window.col_off + final_window.col_off,
                    source_window.row_off + final_window.row_off,
                    final_window.width,
                    final_window.height,
                )
                crop_img = raster.read(
                    indexes=source_raster._bands,
                    masked=source_raster._masked,
                    window=read_window,
                )
            else:
                crop_img = source.read(
                    indexes=source_raster._bands,
                    masked=source_raster._masked,
                    window=final_window_disk,
                    out_shape=(final_window.height, final_window.width),
                )

        # Squeeze first axis for single-band
        if crop_img.ndim == 3 and crop_img.shape[0] == 1:
            crop_img = crop_img.squeeze(axis=0)

        # Restore logical mask values from their on-disk integer representation, keeping missing cells masked
        if source_raster.is_mask:
            crop_img = crop_img.astype(bool)

    return crop_img, tfm


#########
# 4/ CLIP
#########


def _apply_clip_geometry(
    data: NDArrayNum | MArrayNum,
    inside: NDArrayBool,
    nodata: int | float | None,
) -> MArrayNum:
    """Mask cells outside a clipping geometry."""

    outside = ~inside
    if data.ndim == 3 and outside.ndim == 2:
        outside = np.broadcast_to(outside, data.shape)
    combined_mask = np.logical_or(np.ma.getmaskarray(data), outside)
    return np.ma.masked_array(np.ma.getdata(data), mask=combined_mask, fill_value=nodata)


def _multiproc_clip_block(
    source_raster: Raster,
    block_id: dict[str, Any],
    burn: _VectorBurnSpec,
    all_touched: bool,
    output_nodata: int | float | None,
) -> tuple[Raster, tuple[int, int, int, int]]:
    """Read and clip one raster block with its preselected geometries."""

    # Read only this block and rasterize the features selected by the parent spatial index
    pixel_bounds = (block_id["xs"], block_id["ys"], block_id["xe"], block_id["ye"])
    block = source_raster.icrop(bbox=pixel_bounds)
    block_geogrid = GeoGrid(transform=block.transform, shape=block.shape, crs=block.crs)
    inside = _rasterize_selected_on_geogrid(
        block_geogrid,
        burn,
        out_value=0,
        out_dtype=np.uint8,
        all_touched=all_touched,
    ).view(np.bool_)

    # Apply the mask without changing the source grid, values or existing missing cells
    clipped = _apply_clip_geometry(block.data, inside=inside, nodata=output_nodata)
    output = block.copy(new_array=clipped)
    if output.nodata != output_nodata:
        output.set_nodata(output_nodata, update_array=False, update_mask=False)
    if output.is_mask:
        output.astype("uint8", inplace=True)
        output.set_nodata(255)

    # Return the unchanged destination positions for the shared output writer
    destination = (block_id["ys"], block_id["ye"], block_id["xs"], block_id["xe"])
    return output, destination


def _multiproc_clip(
    source_raster: Raster,
    clipping_vector: Vector,
    all_touched: bool,
    output_nodata: int | float | None,
    mp_config: MultiprocConfig,
) -> Raster:
    """Partition clipping geometries, process raster blocks, and write their completed output."""

    # Build one spatial index in the parent and select only the geometries needed by each block
    grid = GeoGrid(transform=source_raster.transform, shape=source_raster.shape, crs=source_raster.crs)
    chunks = normalize_chunks(chunks=_split_chunk_size(mp_config.chunks), shape=source_raster.shape)
    tiling = ChunkedGeoGrid(grid=grid, chunks=chunks)
    block_ids = tiling.get_block_locations()
    block_geogrids = tiling.get_blocks_as_geogrids()
    burn = _normalize_burn_values(clipping_vector.ds.geometry.values, in_value=1)
    block_burns = _partition_burn_by_geogrids(burn, block_geogrids)

    # Send each worker its source window and the much smaller matching geometry subset
    tasks = [
        mp_config.cluster.submit(
            _multiproc_clip_block,
            source_raster,
            block_id,
            block_burn,
            all_touched,
            output_nodata,
        )
        for block_id, block_burn in zip(block_ids, block_burns)
    ]

    # Write completed blocks with the same metadata and logical mask representation as the source
    source_is_mask = source_raster.is_mask
    file_metadata = {
        "width": source_raster.width,
        "height": source_raster.height,
        "count": source_raster.count,
        "crs": source_raster.crs,
        "transform": source_raster.transform,
        "dtype": np.dtype("uint8") if source_is_mask else source_raster.dtype,
        "nodata": 255 if source_is_mask else output_nodata,
    }
    output = _write_multiproc_result(tasks, mp_config, file_metadata, tags=dict(source_raster.tags))
    if output._is_bigtiff():
        warnings.warn(
            "Due to the size of the output raster, it has been saved with a BigTIFF format.",
            category=UserWarning,
        )
    return output


def _clip(
    source_raster: RasterType,
    mask: Any,
    all_touched: bool = False,
    mp_config: MultiprocConfig | None = None,
) -> Any:
    """
    Clip raster cells outside a geometry using eager, Dask or multiprocessing execution.

    _clip_geodataframe() normalizes mask features in the raster CRS without dissolving them. create_mask() partitions
    those features for eager or Dask masking, while _multiproc_clip() selects the matching features before dispatching
    each raster block.
    """

    from geoutils.vector.vector import Vector

    dask_backend = da is not None and source_raster._chunks is not None
    if mp_config is not None and dask_backend:
        raise ValueError(
            "Cannot use Multiprocessing and Dask simultaneously. To use Dask, remove mp_config from clip()."
        )

    # Normalize and reproject clipping features once while keeping them separate for block selection
    target_crs = None if source_raster.crs is None else CRS.from_user_input(source_raster.crs)
    clipping_vector = Vector(_clip_geodataframe(mask, target_crs=target_crs))

    if mp_config is not None:
        if source_raster._is_xr:
            raise ValueError("Multiprocessing clipping requires a Raster input rather than an Xarray accessor.")

        # A file output needs a concrete nodata value to preserve newly clipped cells when it is read again
        output_nodata = source_raster.nodata
        if output_nodata is None and not source_raster.is_mask:
            output_nodata = _default_nodata(source_raster.dtype)
            warnings.warn(f"No nodata value is defined; multiprocessing clip() will use the default {output_nodata}.")
        return _multiproc_clip(
            cast("Raster", source_raster),
            clipping_vector,
            all_touched,
            output_nodata,
            mp_config,
        )

    # Match the source grid and let create_mask() choose eager or Dask execution from the reference
    inside = clipping_vector.create_mask(ref=source_raster, all_touched=all_touched, as_array=True)

    if source_raster._is_xr:
        assert source_raster._obj is not None
        return source_raster._obj.where(inside)

    clipped = _apply_clip_geometry(source_raster.data, inside=inside, nodata=source_raster.nodata)
    return source_raster.copy(new_array=clipped)


##############
# 5/ TRANSLATE
##############


def _translate(
    transform: affine.Affine,
    xoff: float,
    yoff: float,
    distance_unit: Literal["georeferenced", "pixel"] = "georeferenced",
) -> affine.Affine:
    """
    Translate geotransform horizontally, either in pixels or georeferenced units.

    :param transform: Input geotransform.
    :param xoff: Translation x offset.
    :param yoff: Translation y offset.
    :param distance_unit: Distance unit, either 'georeferenced' (default) or 'pixel'.

    :return: Translated transform.
    """

    if distance_unit not in ["georeferenced", "pixel"]:
        raise ValueError("Argument 'distance_unit' should be either 'pixel' or 'georeferenced'.")

    # Get transform
    dx, b, xmin, d, dy, ymax = list(transform)[:6]

    # Convert pixel offsets to georeferenced units
    if distance_unit == "pixel":
        xoff *= dx
        yoff *= dy

    return rio.transform.Affine(dx, b, xmin + xoff, d, dy, ymax + yoff)
