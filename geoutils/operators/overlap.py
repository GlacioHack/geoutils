# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module to estimate fractional overlap area for polygons intersecting partially cells of a regular grid."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import rasterio as rio
import shapely
from numpy.typing import NDArray

from geoutils._misc import import_optional
from geoutils._typing import NDArrayNum
from geoutils.operators.base import LocalData

try:
    from numba import njit
except ImportError:
    njit = None

_jit = njit(cache=True) if njit is not None else lambda function: function

if TYPE_CHECKING:
    from geoutils.operators.reducer import Reducer

# Ignore tiny apparent overlaps caused by rounding when two edges only touch
_MIN_FRACTION = np.finfo(np.float64).eps * 64
OverlapBackend = Literal["auto", "exactextract", "shapely"]
RasterOverlapBackend = Literal["auto", "numba", "exactextract", "shapely"]


######################################
# 1/ INTERSECTED CELLS AND THEIR VALUES
######################################


@dataclass(frozen=True)
class GridIntersection:
    """
    This class describes the raster cells intersected by each geometry.

    Rows, columns and covered fractions are stored in the same order. Each fraction is the area of the intersection
    divided by the cell area (e.g. 0.5 means that half of the cell is covered). Reducers can use these fractions as
    weights.

    Use for_geometry() to read the cells for one geometry, ``offsets`` stores where each geometry starts in the arrays.

    :param offsets: Start positions of each geometry and the final entry count, with length n_geometries + 1.
    :param rows: Grid row for every positive area intersection.
    :param columns: Grid column for every positive area intersection.
    :param fractions: Fraction of the corresponding grid cell covered by the geometry.
    """

    offsets: NDArray[np.int64]
    rows: NDArray[np.int64]
    columns: NDArray[np.int64]
    fractions: NDArray[np.float64]

    def __post_init__(self) -> None:
        """Convert the arrays and check their lengths, offsets, and coverage fractions."""

        # Use integer indexes + floating fractions for all input array types
        offsets = np.asarray(self.offsets, dtype=np.int64)
        rows = np.asarray(self.rows, dtype=np.int64)
        columns = np.asarray(self.columns, dtype=np.int64)
        fractions = np.asarray(self.fractions, dtype=np.float64)

        # Internal checks for inputs
        if offsets.ndim != 1 or len(offsets) == 0 or offsets[0] != 0:
            raise ValueError("GridIntersection offsets must be one-dimensional and start at zero.")
        if rows.ndim != 1 or columns.ndim != 1 or fractions.ndim != 1:
            raise ValueError("GridIntersection cell arrays must be one-dimensional.")
        if len(rows) != len(columns) or len(rows) != len(fractions) or offsets[-1] != len(rows):
            raise ValueError("GridIntersection cell arrays and final offset must have matching lengths.")
        if np.any(np.diff(offsets) < 0):
            raise ValueError("GridIntersection offsets must be non-decreasing.")
        if np.any(~np.isfinite(fractions)) or np.any(fractions <= 0) or np.any(fractions > 1):
            raise ValueError("GridIntersection fractions must be finite and in the interval (0, 1].")

        # This dataclass is frozen, so we assign the converted arrays through object
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "fractions", fractions)

    @property
    def geometry_count(self) -> int:
        """Return the number of geometries represented by the offsets."""

        return len(self.offsets) - 1

    def for_geometry(self, index: int) -> tuple[NDArrayNum, NDArrayNum, NDArrayNum]:
        """Read the raster cells covered by one geometry.

        :param index: Geometry position in the original input sequence (starting at zero).
        :returns: Row indexes, column indexes and covered fractions, in matching 1D arrays.
        """

        if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
            raise TypeError("GridIntersection geometry index must be an integer.")
        if index < 0 or index >= self.geometry_count:
            raise IndexError("GridIntersection geometry index is out of range.")
        start, stop = self.offsets[index : index + 2]
        selection = slice(int(start), int(stop))
        return self.rows[selection], self.columns[selection], self.fractions[selection]


def _union_geometries(geometries: Sequence[Any] | NDArray[Any]) -> Any:
    """Merge geometries, or return an empty geometry collection when none are supplied."""

    geometry_array = np.asarray(geometries, dtype=object).reshape(-1)
    return shapely.GeometryCollection() if len(geometry_array) == 0 else shapely.union_all(geometry_array)


def _union_geometries_by_label(
    geometries: Sequence[Any] | NDArray[Any],
    feature_labels: Sequence[Any],
    output_labels: Sequence[Any],
) -> NDArray[Any]:
    """Combine geometries with each requested label and preserve the label order."""

    geometry_array = np.asarray(geometries, dtype=object).reshape(-1)
    label_values = list(feature_labels)
    if len(geometry_array) != len(label_values):
        raise ValueError("Geometry and feature-label sequences must have the same length.")

    # Add an empty geometry for a missing category so output layers still line up with the requested labels
    grouped_geometries = []
    for output_label in output_labels:
        selected = np.asarray([label == output_label for label in label_values], dtype=bool)
        grouped_geometries.append(_union_geometries(geometry_array[selected]))
    return np.asarray(grouped_geometries, dtype=object)


def _grid_intersection_local_data(
    array: NDArrayNum,
    transform: rio.transform.Affine,
    overlap: GridIntersection,
    *,
    source_id_offset: int = 0,
    source_index_offset: tuple[int, int] = (0, 0),
    source_shape: tuple[int, int] | None = None,
    source_nodata: int | float | None = None,
    eligible: NDArrayNum | None = None,
    targets: NDArrayNum | None = None,
) -> list[LocalData]:
    """
    Collect the values, coordinates + area weights used for a reduction within each geometry.

    GridIntersection.for_geometry() gives us the covered cells. We read their values from the array and record
    any nodata/masked cells in LocalData.valid, leaving the reducer to apply its nodata rule.
    """

    # Read the values and mask separately (masked values must still be reported to the reducer)
    source = np.asanyarray(np.ma.getdata(array))
    if source.ndim != 2:
        raise ValueError("Grid-intersection reduction requires a two-dimensional source array.")
    source_mask = np.ma.getmaskarray(array) if np.ma.isMaskedArray(array) else None
    eligibility = None if eligible is None else np.asarray(eligible, dtype=bool)
    if eligibility is not None and eligibility.shape != source.shape:
        raise ValueError("Grid-intersection eligibility must match the source array shape.")
    if targets is not None and (targets.ndim != 2 or len(targets) != overlap.geometry_count):
        raise ValueError("Grid-intersection targets must contain one coordinate per geometry.")

    # Build one group per geometry, including any invalid cells that it covers
    local_inputs = []
    for geometry_index in range(overlap.geometry_count):
        rows, columns, fractions = overlap.for_geometry(geometry_index)
        values = np.asarray(source[rows, columns])
        valid = np.isfinite(values)
        if source_mask is not None:
            valid &= ~source_mask[rows, columns]
        if source_nodata is not None:
            valid &= values != source_nodata
        if eligibility is not None:
            valid &= eligibility[rows, columns]

        # The same source cell may contribute to several geometries: give it the same ID each time
        # Uncertainty propagation then knows these are repeated uses of one observation
        width = source.shape[1] if source_shape is None else source_shape[1]
        source_ids = source_id_offset + (rows + source_index_offset[0]) * width + columns + source_index_offset[1]
        source_x = transform.a * (columns + 0.5) + transform.b * (rows + 0.5) + transform.c
        source_y = transform.d * (columns + 0.5) + transform.e * (rows + 0.5) + transform.f
        coordinates = np.column_stack((source_x, source_y))
        target = None if targets is None else np.asarray(targets[geometry_index])
        distances = None if target is None else np.linalg.norm(coordinates - target, axis=1)
        local_inputs.append(
            LocalData(
                values=values,
                valid=valid,
                source_ids=source_ids,
                coordinates=coordinates,
                target=target,
                distances=distances,
                support_weights=fractions,
            )
        )
    return local_inputs


###########################
# 2/ GRID GEOMETRY HELPERS
###########################


def _candidate_windows(
    geometries: NDArray[Any],
    transform: rio.transform.Affine,
    shape: tuple[int, int],
) -> tuple[NDArrayNum, NDArrayNum, NDArrayNum, NDArrayNum]:
    """Return a clipped raster window around the bounds of each geometry."""

    geometry_bounds = np.asarray(shapely.bounds(geometries), dtype=np.float64)
    valid = np.all(np.isfinite(geometry_bounds), axis=1) & ~np.asarray(shapely.is_empty(geometries), dtype=bool)
    row_start = np.zeros(len(geometries), dtype=np.int64)
    row_stop = np.zeros(len(geometries), dtype=np.int64)
    column_start = np.zeros(len(geometries), dtype=np.int64)
    column_stop = np.zeros(len(geometries), dtype=np.int64)
    if not np.any(valid):
        return row_start, row_stop, column_start, column_stop

    # We need all four corners of the bounds (with a rotated raster, X/Y both affect the row/column indexes)
    selected_bounds = geometry_bounds[valid]
    min_x, min_y, max_x, max_y = selected_bounds.T
    corner_x = np.column_stack((min_x, max_x, max_x, min_x))
    corner_y = np.column_stack((min_y, min_y, max_y, max_y))
    inverse = ~transform
    pixel_columns = inverse.a * corner_x + inverse.b * corner_y + inverse.c
    pixel_rows = inverse.d * corner_x + inverse.e * corner_y + inverse.f

    # Round outwards to include every possible cell, then clip to the raster bounds
    row_start[valid] = np.maximum(np.floor(np.min(pixel_rows, axis=1)).astype(np.int64), 0)
    row_stop[valid] = np.minimum(np.ceil(np.max(pixel_rows, axis=1)).astype(np.int64), shape[0])
    column_start[valid] = np.maximum(np.floor(np.min(pixel_columns, axis=1)).astype(np.int64), 0)
    column_stop[valid] = np.minimum(np.ceil(np.max(pixel_columns, axis=1)).astype(np.int64), shape[1])
    row_stop = np.maximum(row_stop, row_start)
    column_stop = np.maximum(column_stop, column_start)
    return row_start, row_stop, column_start, column_stop


def _candidate_cells(
    first_geometry: int,
    row_start: NDArrayNum,
    row_stop: NDArrayNum,
    column_start: NDArrayNum,
    column_stop: NDArrayNum,
) -> tuple[NDArrayNum, NDArrayNum, NDArrayNum]:
    """List the cells in each rectangular window, with the geometry index repeated for every cell."""

    # Count cells per window so we can build matching geometry/row/column arrays without a Python loop
    row_counts = row_stop - row_start
    column_counts = column_stop - column_start
    counts = row_counts * column_counts
    local_geometry = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    if len(local_geometry) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty

    # Number cells from zero within each window, one row at a time
    # Dividing by the window width gives the row, remainder gives the column
    starts = np.cumsum(counts, dtype=np.int64) - counts
    local_position = np.arange(np.sum(counts), dtype=np.int64) - np.repeat(starts, counts)
    repeated_column_counts = column_counts[local_geometry]
    rows = row_start[local_geometry] + local_position // repeated_column_counts
    columns = column_start[local_geometry] + local_position % repeated_column_counts
    return local_geometry + first_geometry, rows, columns


def _cell_polygons(rows: NDArrayNum, columns: NDArrayNum, transform: rio.transform.Affine) -> NDArray[Any]:
    """Build Shapely polygons for grid cells in the supplied row and column order."""

    corner_columns = np.column_stack((columns, columns + 1, columns + 1, columns))
    corner_rows = np.column_stack((rows, rows, rows + 1, rows + 1))
    corner_x = transform.a * corner_columns + transform.b * corner_rows + transform.c
    corner_y = transform.d * corner_columns + transform.e * corner_rows + transform.f
    rings = np.stack((corner_x, corner_y), axis=-1)
    return np.asarray(shapely.polygons(rings), dtype=object)


def _assemble_grid_intersection(
    geometry_count: int,
    selected_geometries: list[NDArrayNum],
    selected_rows: list[NDArrayNum],
    selected_columns: list[NDArrayNum],
    selected_fractions: list[NDArrayNum],
) -> GridIntersection:
    """Join the calculated cell groups into one GridIntersection object."""

    # The batches already follow the geometry order, so we can concatenate without sorting
    if selected_rows:
        geometry_ids = np.concatenate(selected_geometries)
        rows = np.asarray(np.concatenate(selected_rows), dtype=np.int64)
        columns = np.asarray(np.concatenate(selected_columns), dtype=np.int64)
        fractions = np.asarray(np.concatenate(selected_fractions), dtype=np.float64)
        counts = np.bincount(geometry_ids, minlength=geometry_count)
    else:
        rows = np.empty(0, dtype=np.int64)
        columns = np.empty(0, dtype=np.int64)
        fractions = np.empty(0, dtype=np.float64)
        counts = np.zeros(geometry_count, dtype=np.int64)
    # Cumulative cell counts mark where each geometry starts + where the last one ends
    offsets = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(counts, dtype=np.int64)))
    return GridIntersection(offsets=offsets, rows=rows, columns=columns, fractions=fractions)


def _pixel_corners_are_aligned(pixel_columns: NDArrayNum, pixel_rows: NDArrayNum) -> bool:
    """Check whether the four corners form rectangles along the source rows/columns."""

    # Opposite edges should have equal row/column coordinates (allowing for transform rounding)
    aligned = np.allclose(pixel_columns[:, 0], pixel_columns[:, 3], rtol=0, atol=1e-10)
    aligned &= np.allclose(pixel_columns[:, 1], pixel_columns[:, 2], rtol=0, atol=1e-10)
    aligned &= np.allclose(pixel_rows[:, 0], pixel_rows[:, 1], rtol=0, atol=1e-10)
    aligned &= np.allclose(pixel_rows[:, 2], pixel_rows[:, 3], rtol=0, atol=1e-10)
    return bool(aligned)


def _corners_are_grid_aligned(corners: NDArrayNum, transform: rio.transform.Affine) -> bool:
    """Check whether four corner shapes follow the grid row/column directions."""

    corner_array = np.asarray(corners, dtype=np.float64)
    if corner_array.ndim != 3 or corner_array.shape[1:] != (4, 2):
        raise ValueError("Grid-intersection corners must have shape (n_geometries, 4, 2).")
    inverse = ~transform
    pixel_columns = inverse.a * corner_array[..., 0] + inverse.b * corner_array[..., 1] + inverse.c
    pixel_rows = inverse.d * corner_array[..., 0] + inverse.e * corner_array[..., 1] + inverse.f
    return _pixel_corners_are_aligned(pixel_columns, pixel_rows)


##################################
# 3/ EXACTEXTRACT INPUTS AND OUTPUTS
##################################


# 3.1/ Match reducers to ExactExtract statistics
##############################################


def _exactextract_operation(operator: Reducer) -> str | None:
    """Return the ExactExtract name for a built-in Reducer with its default behavior."""

    # Reducers import neighborhood selection, which imports this module for grid intersections
    from geoutils.operators.reducer import Maximum, Mean, Minimum, Mode, StandardDeviation, Sum

    # Subclasses may change the calculation, so only these exact classes match an ExactExtract statistic
    if type(operator) is Mean:
        return "mean"
    if type(operator) is Sum:
        return "sum"
    if type(operator) is Minimum:
        return "min"
    if type(operator) is Maximum:
        return "max"
    if type(operator) is StandardDeviation:
        return "stdev"
    if type(operator) is Mode and operator.weighted and operator.tie_break == "largest":
        return "majority"
    return None


# 3.2/ Supply geometries and raster values, and collect results
############################################################


def _run_exactextract(
    geometries: Sequence[Any] | NDArray[Any],
    transform: rio.transform.Affine,
    shape: tuple[int, int],
    *,
    values: NDArrayNum | None = None,
    operation: str | None = None,
) -> Any:
    """
    Use ExactExtract to calculate covered cell fractions or a statistic within each geometry.

    _WKBFeatureSource sends geometries in binary form (WKB), which avoids conversion to/from GeoJSON. To get covered
    fractions, _ConstantRasterSource supplies a raster of ones and _CoverageWriter collects the cell IDs + fractions.
    For a statistic such as mean, NumPyRasterSource supplies the real values and _ScalarWriter collects the results.
    """

    # The classes below inherit from ExactExtract classes, so we define them after this optional import
    import_optional("exactextract", extra_name="exactextract")
    from exactextract import exact_extract
    from exactextract.feature import Feature, FeatureSource
    from exactextract.raster import NumPyRasterSource, RasterSource
    from exactextract.writer import Writer

    # 1/ Send geometries as binary data, in their original order

    class _WKBFeature(Feature):  # type: ignore[misc]
        """Pass one binary geometry to ExactExtract."""

        def __init__(self, geometry: bytes) -> None:
            """Store one geometry without converting it through GeoJSON."""

            super().__init__()
            self._geometry = geometry

        def geometry(self) -> bytes:
            """Return the stored WKB geometry."""

            return self._geometry

        def set_geometry_format(self) -> str:
            """Tell ExactExtract that geometry() returns WKB."""

            return "wkb"

        def fields(self) -> list[str]:
            """Report no attributes (we only need the geometry)."""

            return []

    class _WKBFeatureSource(FeatureSource):  # type: ignore[misc]
        """Read Shapely geometries in the binary format expected by ExactExtract."""

        def __init__(self, geometries: Sequence[Any] | NDArray[Any]) -> None:
            """Convert geometries to WKB once without changing their order."""

            super().__init__()
            self._geometries = np.asarray(shapely.to_wkb(geometries), dtype=object)

        def count(self) -> int:
            """Return the number of source geometries."""

            return len(self._geometries)

        def __iter__(self) -> Iterator[_WKBFeature]:
            """Yield the WKB geometries as ExactExtract features in source order."""

            return (_WKBFeature(geometry) for geometry in self._geometries)

        def srs_wkt(self) -> None:
            """Omit the CRS (the geometry coordinates already match the raster's CRS)."""

            return None

    # 2/ Read result fields directly into a dictionary (no geometry conversion is needed)

    class _ResultFeature(Feature):  # type: ignore[misc]
        """Read one ExactExtract result as a dictionary."""

        def __init__(self) -> None:
            """Start with an empty result dictionary."""

            super().__init__()
            self.values: dict[str, Any] = {}

        def set(self, name: str, value: Any) -> None:
            """Store a calculated value under the name provided by ExactExtract."""

            self.values[name] = value

        def get(self, name: str) -> Any:
            """Return one stored value."""

            return self.values[name]

        def fields(self) -> list[str]:
            """Return stored field names."""

            return list(self.values)

        def geometry(self) -> None:
            """Return no geometry (only calculated values are requested)."""

            return None

        def set_geometry(self, geometry: Any) -> None:
            """Ignore output geometries, as we already have the input ones."""

        def set_geometry_format(self) -> str:
            """Provide the geometry format required by the Feature interface."""

            return "wkb"

    # 3/ For cell coverage, provide a raster of ones without allocating the complete array

    class _ConstantRasterSource(RasterSource):  # type: ignore[misc]
        """Provide an array of ones for each raster window requested by ExactExtract."""

        def __init__(self, shape: tuple[int, int], extent: tuple[float, float, float, float]) -> None:
            """Store the grid shape and bounds without allocating a complete raster."""

            super().__init__()
            self._shape = shape
            self._extent = extent

        def res(self) -> tuple[float, float]:
            """Return positive X/Y cell sizes."""

            return (
                (self._extent[2] - self._extent[0]) / self._shape[1],
                (self._extent[3] - self._extent[1]) / self._shape[0],
            )

        def extent(self) -> tuple[float, float, float, float]:
            """Return left, bottom, right and top bounds."""

            return self._extent

        def nodata_value(self) -> None:
            """Mark every generated grid cell as valid."""

            return None

        def read_window(self, x0: int, y0: int, nx: int, ny: int) -> NDArrayNum:
            """Create a window of ones (every cell is valid)."""

            return np.ones((ny, nx), dtype=np.uint8)

        def srs_wkt(self) -> None:
            """Return no CRS because geometry coordinates already match the grid."""

            return None

    # 4/ Collect either covered fractions or one statistic per input geometry

    class _CoverageWriter(Writer):  # type: ignore[misc]
        """Collect cell IDs + covered fractions as arrays, one pair per geometry."""

        def __init__(self) -> None:
            """Create the output lists in feature order."""

            super().__init__()
            self.cell_ids: list[NDArrayNum] = []
            self.coverages: list[NDArrayNum] = []

        def add_operation(self, operation: Any) -> None:
            """Accept an operation from ExactExtract (the two output names are already known)."""

        def write(self, feature: Any) -> None:
            """Read the cell IDs and covered fractions for one geometry."""

            output = _ResultFeature()
            feature.copy_to(output)
            cell_ids = output.values.get("cell_id")
            coverages = output.values.get("coverage")
            self.cell_ids.append(
                np.empty(0, dtype=np.int64) if cell_ids is None else np.asarray(cell_ids, dtype=np.int64)
            )
            self.coverages.append(
                np.empty(0, dtype=np.float64) if coverages is None else np.asarray(coverages, dtype=np.float64)
            )

        def features(self) -> tuple[list[NDArrayNum], list[NDArrayNum]]:
            """Return cell IDs and coverage fractions in feature order."""

            return self.cell_ids, self.coverages

    class _ScalarWriter(Writer):  # type: ignore[misc]
        """Collect one scalar ExactExtract result per feature."""

        def __init__(self) -> None:
            """Start a result list: ExactExtract will supply the field name before writing values."""

            super().__init__()
            self.field_name: str | None = None
            self.values: list[float] = []

        def add_operation(self, operation: Any) -> None:
            """Record which field contains the requested statistic."""

            self.field_name = operation.name

        def write(self, feature: Any) -> None:
            """Read one calculated value, using NaN when the geometry has no result."""

            output = _ResultFeature()
            feature.copy_to(output)
            value = (
                output.values.get(self.field_name)
                if self.field_name is not None
                else next(iter(output.values.values()), None)
            )
            self.values.append(np.nan if value is None else float(value))

        def features(self) -> NDArrayNum:
            """Return one floating-point result per feature."""

            return np.asarray(self.values, dtype=np.float64)

    # 5/ Check the grid, choose the raster/collector pair, then run ExactExtract

    # ExactExtract expects an unrotated raster with north upwards, and bounds in left/bottom/right/top order
    if transform.b != 0 or transform.d != 0 or transform.a <= 0 or transform.e >= 0:
        raise ValueError("ExactExtract requires a north-up grid with positive X and negative Y resolution.")
    left = transform.c
    top = transform.f
    right = left + transform.a * shape[1]
    bottom = top + transform.e * shape[0]
    extent = (float(left), float(bottom), float(right), float(top))

    # Covered fractions need every cell to be valid; statistics use the supplied raster values + their nodata
    if values is None:
        raster_source: RasterSource = _ConstantRasterSource(shape, extent)
        operations = ["cell_id", "coverage"]
        writer: Writer = _CoverageWriter()
    else:
        if values.ndim != 2:
            raise ValueError("ExactExtract reduction requires a two-dimensional value array.")
        if operation is None:
            raise ValueError("An ExactExtract reduction requires an operation name.")
        # ExactExtract does not consistently honor NumPy masks, so pass missing cells as an explicit nodata value
        if np.ma.isMaskedArray(values):
            floating_values = values if np.issubdtype(values.dtype, np.floating) else values.astype(np.float64)
            raster_values = np.ma.filled(floating_values, np.nan)
        else:
            raster_values = values
        nodata = np.nan if np.issubdtype(raster_values.dtype, np.floating) else None
        raster_source = NumPyRasterSource(raster_values, *extent, nodata=nodata)
        operations = [operation]
        writer = _ScalarWriter()

    # With many geometries, we read each raster window once and process all geometries that touch it
    # For a small geometry list, ExactExtract works through the geometries one at a time
    strategy = "raster-sequential" if len(geometries) >= 64 else "feature-sequential"
    return exact_extract(raster_source, _WKBFeatureSource(geometries), operations, strategy=strategy, output=writer)


##################################
# 4/ CALCULATE OVERLAPPING AREAS
##################################


# 4.1/ Cell coverage with Shapely
###############################


def _grid_intersection_fractions_shapely(
    geometry_array: NDArray[Any],
    transform: rio.transform.Affine,
    shape: tuple[int, int],
    *,
    cell_area: float,
    batch_size: int,
) -> GridIntersection:
    """Intersect polygons with affine grid cells in batches and divide each intersection area by the cell area."""

    # For Shapely, start with a clipped raster window around each geometry
    row_start, row_stop, column_start, column_stop = _candidate_windows(geometry_array, transform, shape)
    selected_geometries: list[NDArrayNum] = []
    selected_rows: list[NDArrayNum] = []
    selected_columns: list[NDArrayNum] = []
    selected_fractions: list[NDArrayNum] = []

    for first in range(0, len(geometry_array), batch_size):
        # Build cell polygons for one batch at a time (a complete raster's worth would use too much memory)
        stop = min(first + batch_size, len(geometry_array))
        geometry_ids, rows, columns = _candidate_cells(
            first,
            row_start[first:stop],
            row_stop[first:stop],
            column_start[first:stop],
            column_stop[first:stop],
        )
        if len(rows) == 0:
            continue

        # Divide the intersection area by the cell area; touching an edge or corner contributes no area
        cells = _cell_polygons(rows, columns, transform)
        intersections = shapely.intersection(cells, geometry_array[geometry_ids])
        fractions = np.asarray(shapely.area(intersections), dtype=np.float64) / cell_area
        positive = fractions > _MIN_FRACTION
        selected_geometries.append(geometry_ids[positive])
        selected_rows.append(rows[positive])
        selected_columns.append(columns[positive])
        selected_fractions.append(np.minimum(fractions[positive], 1.0))

    return _assemble_grid_intersection(
        len(geometry_array),
        selected_geometries,
        selected_rows,
        selected_columns,
        selected_fractions,
    )


# 4.2/ Cell coverage with ExactExtract
####################################


def _grid_intersection_fractions_exactextract(
    geometry_array: NDArray[Any],
    transform: rio.transform.Affine,
    shape: tuple[int, int],
    *,
    usable: NDArray[np.bool_],
) -> GridIntersection:
    """Read ExactExtract cell IDs and fractions on a north-up grid, leaving empty groups for unusable geometries."""

    # Read only usable geometries, then restore their original positions in the output groups
    selected_positions = np.flatnonzero(usable)
    if len(selected_positions) == 0:
        return _assemble_grid_intersection(len(geometry_array), [], [], [], [])
    cell_ids, coverage_arrays = _run_exactextract(geometry_array[selected_positions], transform, shape)
    selected_geometries: list[NDArrayNum] = []
    selected_rows: list[NDArrayNum] = []
    selected_columns: list[NDArrayNum] = []
    selected_fractions: list[NDArrayNum] = []

    # ExactExtract numbers cells in row order: divide by raster width for rows, use the remainder for columns
    for geometry_id, ids, fractions in zip(selected_positions, cell_ids, coverage_arrays):
        positive = np.isfinite(fractions) & (fractions > _MIN_FRACTION)
        ids = ids[positive]
        selected_geometries.append(np.full(len(ids), geometry_id, dtype=np.int64))
        selected_rows.append(ids // shape[1])
        selected_columns.append(ids % shape[1])
        selected_fractions.append(np.minimum(fractions[positive], 1.0))

    return _assemble_grid_intersection(
        len(geometry_array),
        selected_geometries,
        selected_rows,
        selected_columns,
        selected_fractions,
    )


# 4.3/ Polygon intersections: validate inputs and select a backend
################################################################


def _grid_intersection_fractions(
    geometries: Sequence[Any] | NDArray[Any],
    transform: rio.transform.Affine,
    shape: tuple[int, int],
    *,
    batch_size: int = 4096,
    backend: OverlapBackend = "auto",
) -> GridIntersection:
    """
    Intersect geometries with a regular grid and return the covered cells and area fractions.

    With backend="auto", we use ExactExtract when it is installed and the grid is north-up (no rotation or shear,
    positive X resolution and negative Y resolution). Otherwise we use Shapely. An explicit backend="shapely"
    always uses Shapely; backend="exactextract" requires the optional package and a compatible grid.

    _grid_intersection_fractions_exactextract() calls _run_exactextract() to obtain cell IDs and coverage fractions,
    then converts the IDs to rows and columns. _grid_intersection_fractions_shapely() uses _candidate_windows()
    and _candidate_cells() to select cells, builds their polygons with _cell_polygons(), and divides each Shapely
    intersection area by the cell area. This also works with rotated or sheared grids.

    Both backends discard edge/corner contacts and tiny overlaps caused by rounding. _assemble_grid_intersection()
    groups the remaining cells in input geometry order, including empty groups for geometries covering no cells.
    Coverage is the fraction of each grid cell covered by a geometry, not the fraction of the geometry in that cell.

    We calculate only fractional area from geometry here. Reprojection, rasterization and polygon statistics can all
    use these same fractions.

    :param geometries: Ordered polygonal geometries expressed in the grid coordinate reference system.
    :param transform: Affine mapping from grid-cell corners to geometry coordinates.
    :param shape: Grid height and width.
    :param batch_size: Maximum number of input geometries expanded in one Shapely batch.
    :param backend: Library used to calculate intersections. Auto uses ExactExtract on a compatible grid when it is
        installed, and otherwise uses Shapely.
    :returns: Rows, columns, and covered cell fractions grouped by input geometry.
    """

    # 1/ Validate inputs and identify geometries with usable bounds

    # Check the grid dimensions, batch size and requested library before expanding geometry bounds to cells
    if len(shape) != 2 or shape[0] < 0 or shape[1] < 0:
        raise ValueError("Grid shape must contain non-negative height and width.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("Grid-intersection batch size must be a positive integer.")
    if backend not in {"auto", "exactextract", "shapely"}:
        raise ValueError("Grid-intersection backend must be 'auto', 'exactextract' or 'shapely'.")
    # The transform determinant gives the cell area, including for rotated grids
    cell_area = abs(transform.a * transform.e - transform.b * transform.d)
    if not np.isfinite(cell_area) or cell_area <= 0:
        raise ValueError("Grid transform must describe cells with a positive finite area.")

    # Empty/missing geometries will have no covered cells, but must still occupy a position in the result
    geometry_array = np.asarray(geometries, dtype=object).reshape(-1)
    geometry_bounds = np.asarray(shapely.bounds(geometry_array), dtype=np.float64).reshape((-1, 4))
    usable = ~np.asarray(shapely.is_empty(geometry_array), dtype=bool)
    usable &= ~np.asarray(shapely.is_missing(geometry_array), dtype=bool)
    usable &= np.all(np.isfinite(geometry_bounds), axis=1)

    # 2/ Select ExactExtract for compatible grids or use Shapely

    # ExactExtract accepts north-up grids only, Shapely also handles rotated grids
    exactextract_compatible = transform.b == 0 and transform.d == 0 and transform.a > 0 and transform.e < 0
    exactextract_available = find_spec("exactextract") is not None
    if backend == "exactextract" and not exactextract_available:
        import_optional("exactextract", extra_name="exactextract")
    use_exactextract = exactextract_compatible and exactextract_available and backend != "shapely"
    if use_exactextract:
        return _grid_intersection_fractions_exactextract(geometry_array, transform, shape, usable=usable)
    if backend == "exactextract":
        raise ValueError("ExactExtract requires a north-up grid with positive X and negative Y resolution.")

    return _grid_intersection_fractions_shapely(
        geometry_array, transform, shape, cell_area=cell_area, batch_size=batch_size
    )


# 4.4/ Cell corners: direct overlap for aligned cells or polygon intersections
###########################################################################


@_jit
def _pixel_square_overlap(
    corners: NDArrayNum, row: int, column: int, polygon: NDArrayNum, scratch: NDArrayNum
) -> float:
    """Clip a quadrilateral to one unit source cell and return the covered fraction."""

    polygon[:4] = corners
    count = 4

    # Clip against the left, right, upper and lower sides of the source cell
    for side in range(4):
        axis = 0 if side < 2 else 1
        bound = (column if side == 0 else column + 1) if axis == 0 else (row if side == 2 else row + 1)
        lower = side == 0 or side == 2
        next_count = 0
        for index in range(count):
            previous_index = count - 1 if index == 0 else index - 1
            previous_x = polygon[previous_index, 0]
            previous_y = polygon[previous_index, 1]
            current_x = polygon[index, 0]
            current_y = polygon[index, 1]
            previous_coordinate = previous_x if axis == 0 else previous_y
            current_coordinate = current_x if axis == 0 else current_y
            previous_inside = previous_coordinate >= bound if lower else previous_coordinate <= bound
            current_inside = current_coordinate >= bound if lower else current_coordinate <= bound
            if previous_inside != current_inside:
                fraction = (bound - previous_coordinate) / (current_coordinate - previous_coordinate)
                scratch[next_count, 0] = previous_x + fraction * (current_x - previous_x)
                scratch[next_count, 1] = previous_y + fraction * (current_y - previous_y)
                next_count += 1
            if current_inside:
                scratch[next_count, 0] = current_x
                scratch[next_count, 1] = current_y
                next_count += 1
        polygon, scratch = scratch, polygon
        count = next_count
        if count == 0:
            return 0.0

    # Shoelace area is the overlap fraction because each source cell has unit area
    signed_area = 0.0
    for index in range(1, count - 1):
        first_x = polygon[index, 0] - polygon[0, 0]
        first_y = polygon[index, 1] - polygon[0, 1]
        second_x = polygon[index + 1, 0] - polygon[0, 0]
        second_y = polygon[index + 1, 1] - polygon[0, 1]
        signed_area += first_x * second_y - second_x * first_y
    return min(abs(signed_area) * 0.5, 1.0)


@_jit
def _pixel_quadrilateral_intersections(
    columns: NDArrayNum, rows: NDArrayNum, height: int, width: int
) -> tuple[NDArrayNum, NDArrayNum, NDArrayNum, NDArrayNum]:
    """Collect source cells and areas intersected by quadrilaterals in pixel coordinates."""

    geometry_count = len(columns)
    starts = np.zeros(geometry_count, dtype=np.int64)
    stops = np.zeros(geometry_count, dtype=np.int64)
    lefts = np.zeros(geometry_count, dtype=np.int64)
    rights = np.zeros(geometry_count, dtype=np.int64)
    capacity = 0

    # Bound each quadrilateral in source pixels to size the candidate cell arrays
    for target in range(geometry_count):
        min_row = np.inf
        max_row = -np.inf
        min_column = np.inf
        max_column = -np.inf
        valid = True
        for vertex in range(4):
            row = rows[target, vertex]
            column = columns[target, vertex]
            if not np.isfinite(row) or not np.isfinite(column):
                valid = False
                break
            if row < min_row:
                min_row = row
            if row > max_row:
                max_row = row
            if column < min_column:
                min_column = column
            if column > max_column:
                max_column = column
        if not valid:
            continue
        starts[target] = max(0, int(np.floor(min_row)))
        stops[target] = min(height, int(np.ceil(max_row)))
        lefts[target] = max(0, int(np.floor(min_column)))
        rights[target] = min(width, int(np.ceil(max_column)))
        capacity += max(0, stops[target] - starts[target]) * max(0, rights[target] - lefts[target])

    output_rows = np.empty(capacity, dtype=np.int64)
    output_columns = np.empty(capacity, dtype=np.int64)
    output_fractions = np.empty(capacity, dtype=np.float64)
    offsets = np.zeros(geometry_count + 1, dtype=np.int64)
    corners = np.empty((4, 2), dtype=np.float64)
    polygon = np.empty((12, 2), dtype=np.float64)
    scratch = np.empty((12, 2), dtype=np.float64)
    count = 0

    # Clip only candidate cells, recording positive overlap areas in destination order
    for target in range(geometry_count):
        for vertex in range(4):
            corners[vertex, 0] = columns[target, vertex]
            corners[vertex, 1] = rows[target, vertex]
        for row in range(starts[target], stops[target]):
            for column in range(lefts[target], rights[target]):
                fraction = _pixel_square_overlap(corners, row, column, polygon, scratch)
                if fraction > _MIN_FRACTION:
                    output_rows[count] = row
                    output_columns[count] = column
                    output_fractions[count] = fraction
                    count += 1
        offsets[target + 1] = count
    return offsets, output_rows[:count], output_columns[:count], output_fractions[:count]


def _grid_intersection_fractions_from_corners(
    corners: NDArrayNum,
    transform: rio.transform.Affine,
    shape: tuple[int, int],
    *,
    batch_size: int = 4096,
    backend: RasterOverlapBackend = "auto",
) -> GridIntersection:
    """Calculate covered fractions from the four corners of each destination cell.

    For cells aligned with the source grid, we multiply the row and column overlap lengths. Otherwise Numba clips
    each quadrilateral, or ExactExtract and Shapely intersect polygons on a unit-pixel grid. An affine change
    preserves covered fractions, so all three methods work with rotated or sheared source grids. Numba handles only
    four-corner raster footprints; general polygons use _grid_intersection_fractions().

    :param corners: Four ordered coordinate pairs per footprint, with shape (n_geometries, 4, 2).
    :param transform: Affine mapping from source grid-cell corners to footprint coordinates.
    :param shape: Source grid height and width.
    :param batch_size: Maximum number of aligned footprints expanded in one batch.
    :param backend: Method for unaligned footprints. Auto uses Numba for fewer than 100,000 cells when installed,
        then ExactExtract when installed, or Shapely. Other values select one explicitly. Aligned footprints always
        use direct overlap lengths.
    :returns: Source rows, columns and covered fractions grouped by destination cell.
    """

    # 1/ Validate corners and use polygon intersections when they do not follow the source grid

    corner_array = np.asarray(corners, dtype=np.float64)
    if corner_array.ndim != 3 or corner_array.shape[1:] != (4, 2):
        raise ValueError("Grid-intersection corners must have shape (n_geometries, 4, 2).")
    if backend not in ("auto", "numba", "exactextract", "shapely"):
        raise ValueError("Raster overlap backend must be 'auto', 'numba', 'exactextract' or 'shapely'.")

    # Work in source pixel coordinates: two rotated grids can still follow the same row/column directions
    inverse = ~transform
    pixel_columns = inverse.a * corner_array[..., 0] + inverse.b * corner_array[..., 1] + inverse.c
    pixel_rows = inverse.d * corner_array[..., 0] + inverse.e * corner_array[..., 1] + inverse.f
    # Snap transformed grid lines back to integer pixels before tiny rounding errors create extra cells
    pixel_columns = np.where(
        np.abs(pixel_columns - np.rint(pixel_columns)) < 1e-8, np.rint(pixel_columns), pixel_columns
    )
    pixel_rows = np.where(np.abs(pixel_rows - np.rint(pixel_rows)) < 1e-8, np.rint(pixel_rows), pixel_rows)
    aligned = _pixel_corners_are_aligned(pixel_columns, pixel_rows)
    if not aligned:
        if backend == "numba" and njit is None:
            raise ImportError("Numba overlap requires numba.")
        if backend == "numba" or (backend == "auto" and njit is not None and len(corner_array) < 100_000):
            offsets, rows, columns, fractions = _pixel_quadrilateral_intersections(
                pixel_columns, pixel_rows, shape[0], shape[1]
            )
            return GridIntersection(offsets, rows, columns, fractions)

        # Affine changes preserve covered fractions, so both polygon libraries use the same unit-pixel grid
        virtual_corners = np.stack((pixel_columns, shape[0] - pixel_rows), axis=-1)
        virtual_transform = rio.transform.from_origin(0, shape[0], 1, 1)
        polygons = np.asarray(shapely.polygons(virtual_corners), dtype=object)
        polygon_backend: OverlapBackend = "auto"
        if backend == "exactextract":
            polygon_backend = "exactextract"
        elif backend == "shapely":
            polygon_backend = "shapely"
        return _grid_intersection_fractions(
            polygons, virtual_transform, shape, batch_size=batch_size, backend=polygon_backend
        )

    # 2/ Calculate overlap lengths for aligned cells

    # Round each destination cell's bounds outwards, then clip the rows/columns to the source raster
    left = np.min(pixel_columns, axis=1)
    right = np.max(pixel_columns, axis=1)
    top = np.min(pixel_rows, axis=1)
    bottom = np.max(pixel_rows, axis=1)
    row_start = np.maximum(np.floor(top).astype(np.int64), 0)
    row_stop = np.minimum(np.ceil(bottom).astype(np.int64), shape[0])
    column_start = np.maximum(np.floor(left).astype(np.int64), 0)
    column_stop = np.minimum(np.ceil(right).astype(np.int64), shape[1])
    row_stop = np.maximum(row_stop, row_start)
    column_stop = np.maximum(column_stop, column_start)

    selected_geometries: list[NDArrayNum] = []
    selected_rows: list[NDArrayNum] = []
    selected_columns: list[NDArrayNum] = []
    selected_fractions: list[NDArrayNum] = []
    for first in range(0, len(corner_array), batch_size):
        # List the source cells to check for this batch of destination cells
        stop = min(first + batch_size, len(corner_array))
        geometry_ids, rows, columns = _candidate_cells(
            first,
            row_start[first:stop],
            row_stop[first:stop],
            column_start[first:stop],
            column_stop[first:stop],
        )
        if len(rows) == 0:
            continue
        # Each source cell is 1 x 1 in pixel coordinates, so row overlap * column overlap is its covered fraction
        row_overlap = np.minimum(bottom[geometry_ids], rows + 1) - np.maximum(top[geometry_ids], rows)
        column_overlap = np.minimum(right[geometry_ids], columns + 1) - np.maximum(left[geometry_ids], columns)
        fractions = row_overlap * column_overlap
        positive = fractions > _MIN_FRACTION
        selected_geometries.append(geometry_ids[positive])
        selected_rows.append(rows[positive])
        selected_columns.append(columns[positive])
        selected_fractions.append(np.minimum(fractions[positive], 1.0))

    return _assemble_grid_intersection(
        len(corner_array),
        selected_geometries,
        selected_rows,
        selected_columns,
        selected_fractions,
    )


def _gdal_rectangle_weights(
    overlap: GridIntersection,
    left: NDArrayNum,
    right: NDArrayNum,
    top: NDArrayNum,
    bottom: NDArrayNum,
    shape: tuple[int, int],
) -> GridIntersection:
    """Match GDAL's relative rectangle weights at the edge of a source raster.

    GDAL clips the source cell indexes to the raster but calculates boundary weights from the original, unclipped
    rectangle. An edge cell can therefore have a weight above one. Scale each destination group's weights together
    so they fit GridIntersection's fraction range without changing weighted means or root mean squares.
    """

    # Repeat each destination ID once per contributing source cell
    target_ids = np.repeat(np.arange(overlap.geometry_count), np.diff(overlap.offsets))
    if len(target_ids) == 0:
        return overlap
    columns = overlap.columns
    rows = overlap.rows

    # GDAL gives a single cell along an axis unit weight, even when it covers only part of the rectangle
    first_column = np.maximum(np.floor(left + 1e-10).astype(np.int64), 0)[target_ids]
    last_column = np.minimum(np.ceil(right - 1e-10).astype(np.int64), shape[1])[target_ids]
    first_row = np.maximum(np.floor(top + 1e-10).astype(np.int64), 0)[target_ids]
    last_row = np.minimum(np.ceil(bottom - 1e-10).astype(np.int64), shape[0])[target_ids]
    column_weights = np.where(
        columns == first_column,
        np.where(last_column == first_column + 1, 1.0, 1.0 - (left[target_ids] - first_column)),
        np.where(columns == last_column - 1, 1.0 - (last_column - right[target_ids]), 1.0),
    )
    row_weights = np.where(
        rows == first_row,
        np.where(last_row == first_row + 1, 1.0, 1.0 - (top[target_ids] - first_row)),
        np.where(rows == last_row - 1, 1.0 - (last_row - bottom[target_ids]), 1.0),
    )

    # GDAL can assign more than unit weight to an edge cell when the rectangle extends past the source
    weights = column_weights * row_weights
    group_maximum = np.zeros(overlap.geometry_count, dtype=np.float64)
    np.maximum.at(group_maximum, target_ids, weights)
    fractions = weights / np.maximum(group_maximum[target_ids], 1.0)
    return GridIntersection(overlap.offsets, rows, columns, fractions)
