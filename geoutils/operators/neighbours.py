# Copyright (c) 2026 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Select neighboring points or grid cells for interpolation and reduction."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import shapely
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from geoutils._typing import NDArrayBool, NDArrayNum, Number
from geoutils.operators.base import LocalData
from geoutils.operators.nodata import NodataHandling
from geoutils.operators.overlap import (
    _grid_intersection_fractions,
    _grid_intersection_fractions_from_corners,
    _grid_intersection_local_data,
)
from geoutils.projtools import _affine_matmul

if TYPE_CHECKING:
    import geopandas as gpd
    import rasterio as rio

    from geoutils.operators.interpolator import Interpolator, Kriging, ScipyInterpolationMethod
    from geoutils.operators.nodata import NodataPropagation
    from geoutils.operators.reducer import Reducer


##########################
# 1/ NEIGHBOUR DEFINITIONS
##########################

GridCoverage = Literal["center", "all_touched", "fractional"]


class _DefaultNeighbour(Enum):
    """Distinguish an omitted neighbor limit from an explicit unlimited value."""

    VALUE = 0


@dataclass(frozen=True)
class PointNeighbours:
    """
    Define a point neighborhood by a max number of points inside a radius.

    Pass a PointNeighbours to an Interpolator/Reducer to change which nearby points it uses. For example,
    ``InverseDistance(neighborhood=PointNeighbours(k=12, radius=50))`` uses up to 12 points within 50 coordinate units.
    Distances are measured in X/Y, so the radius defines a circle around each requested position.
    A Reducer can use the same neighborhood when grid() combines nearby point values.

    .. code-block:: python

        from geoutils.operators import PointNeighbours
        from geoutils.operators.interpolator import InverseDistance
        from geoutils.operators.reducer import Mean

        nearby = PointNeighbours(k=12, radius=50)
        gridded = points.grid(ref=reference_raster, resampling=InverseDistance(neighborhood=nearby))
        averaged = points.grid(ref=reference_raster, resampling=Mean(neighborhood=nearby))

    :param k: Maximum number of nearest observations.
    :param radius: Maximum distance from the target in coordinate units.
    """

    k: int | None = None
    radius: float | None = None

    def __post_init__(self) -> None:
        """Check that at least one valid limit was given."""

        if self.k is None and self.radius is None:
            raise ValueError("PointNeighbours requires at least one of k or radius.")
        if self.k is not None and (isinstance(self.k, bool) or not isinstance(self.k, (int, np.integer)) or self.k < 1):
            raise ValueError("PointNeighbours k must be a positive integer.")
        if self.radius is not None and (
            isinstance(self.radius, (bool, np.bool_)) or not np.isfinite(self.radius) or self.radius < 0
        ):
            raise ValueError("PointNeighbours radius must be finite and non-negative.")


@dataclass(frozen=True, init=False)
class GridNeighbours:
    """
    Define a grid neighbourhood for source cells, i.e. a window surrounding the target cells.

    Pass size for a square or circular window, or supply row/column offsets directly to define any window shape.
    For example, ``InverseDistance(neighborhood=GridNeighbours(size=5, shape="circular"))`` uses cells whose
    centers lie within 2.5 pixels of the target cell's center. For a Reducer with ``coverage="fractional"``, a
    circle of the same radius is centered on the requested point.

    For use with a Reducer with either a non-square window or involving area deformations through a CRS change (i.e.
    use in ``reproject()``), exact area coverage can be defined according to the following options:

    - ``center`` uses all cells whose center is intersected by the window shape,
    - ``all_touched`` includes all touched cells,
    - ``fractional`` weights each cell by its area covered fraction, requiring a Reducer that supports weighting.

    For built-in regular interpolation, offsets describe the method's natural stencil: nearest uses (0, 0)
    relative to the nearest cell; linear/slinear use (0, 1) on each axis relative to the lower grid index, and
    PCHIP uses (-1, 0, 1, 2). Extra offsets warn and are ignored; missing offsets raise an error. Cubic, quintic
    and splinef2d fit the complete raster and do not accept GridNeighbours.

    .. code-block:: python

        from geoutils.operators import GridNeighbours
        from geoutils.operators.reducer import Mean

        window = GridNeighbours(size=5, shape="circular")
        values = raster.resample_at_points((x, y), Mean(neighborhood=window), as_array=True)

    :param offsets: Optional row/column offsets relative to the target's containing source cell, except for the
        built-in regular interpolation stencils described above. An integer instead specifies the window size.
    :param window_shape: Shape recorded for an existing set of offsets.
    :param size: Positive odd number of rows and columns in a centered window, used instead of offsets.
    :param shape: Include the whole square or only cells inside its centered circle when size is given.
    :param coverage: How a cells at the edge of a r window.
    """

    offsets: tuple[tuple[int, int], ...]
    window_shape: Literal["square", "circular"] | None = field(default=None, compare=False)
    coverage: GridCoverage = "center"

    def __init__(
        self,
        offsets: tuple[tuple[int, int], ...] | int | None = None,
        window_shape: Literal["square", "circular"] | None = None,
        *,
        size: int | None = None,
        shape: Literal["square", "circular"] = "square",
        coverage: GridCoverage = "center",
    ) -> None:
        """Select a window by size or offsets and choose how source cells contribute."""

        if isinstance(offsets, (int, np.integer)) and size is None:
            size = offsets
            offsets = None
        if size is not None:
            if offsets is not None or window_shape is not None:
                raise ValueError("GridNeighbours accepts either a window size or explicit offsets.")
            if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size < 1 or size % 2 != 1:
                raise ValueError("GridNeighbours window size must be a positive odd integer.")
            if shape not in ("square", "circular"):
                raise ValueError("GridNeighbours window shape must be 'square' or 'circular'.")

            # Select the centered square, excluding corners outside the circle when requested
            half_size = size // 2
            radius = size / 2
            offsets = tuple(
                (row, col)
                for row in range(-half_size, half_size + 1)
                for col in range(-half_size, half_size + 1)
                if shape == "square" or row * row + col * col <= radius * radius
            )
            window_shape = shape
        elif offsets is None:
            raise ValueError("GridNeighbours requires a window size or explicit offsets.")
        elif shape != "square":
            raise ValueError("GridNeighbours shape requires a window size.")

        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "window_shape", window_shape)
        object.__setattr__(self, "coverage", coverage)
        self.__post_init__()

    def __post_init__(self) -> None:
        """Check that the offsets are unique pairs of integers."""

        if len(self.offsets) == 0:
            raise ValueError("GridNeighbours requires at least one source-grid offset.")
        if any(len(offset) != 2 for offset in self.offsets):
            raise ValueError("GridNeighbours offsets must be unique row/column pairs.")
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            for offset in self.offsets
            for value in offset
        ):
            raise ValueError("GridNeighbours offsets must contain integers.")
        normalized = tuple((int(offset[0]), int(offset[1])) for offset in self.offsets)
        if len(set(normalized)) != len(normalized):
            raise ValueError("GridNeighbours offsets must be unique row/column pairs.")
        if self.window_shape not in (None, "square", "circular"):
            raise ValueError("GridNeighbours window shape must be 'square' or 'circular'.")
        if self.coverage not in ("center", "all_touched", "fractional"):
            raise ValueError("GridNeighbours coverage must be 'center', 'all_touched' or 'fractional'.")
        object.__setattr__(self, "offsets", normalized)

    @property
    def overlap(self) -> tuple[int, int]:
        """Return how many extra rows/columns are needed on either side of a raster chunk."""

        return (
            max(abs(row) for row, _ in self.offsets),
            max(abs(col) for _, col in self.offsets),
        )


def _configure_grid_neighbours(
    neighborhood: GridNeighbours | None,
    *,
    size: int | None = None,
    shape: Literal["square", "circular"] | None = None,
    coverage: GridCoverage | None = None,
    default_size: int | None = 3,
) -> GridNeighbours | None:
    """Apply call-level window options without changing the supplied neighborhood."""

    selected_coverage = (
        coverage if coverage is not None else (neighborhood.coverage if neighborhood is not None else "center")
    )
    if size is None and shape is None and neighborhood is not None:
        if selected_coverage == neighborhood.coverage:
            return neighborhood
        return GridNeighbours(neighborhood.offsets, window_shape=neighborhood.window_shape, coverage=selected_coverage)
    if size is None and shape is None and default_size is None:
        return None

    # An explicit size or shape replaces offset patterns with a square or circular window
    window_size = (
        size if size is not None else (default_size if neighborhood is None else 2 * max(neighborhood.overlap) + 1)
    )
    assert window_size is not None
    window_shape = shape or (neighborhood.window_shape if neighborhood is not None else None) or "square"
    return GridNeighbours(size=window_size, shape=window_shape, coverage=selected_coverage)


######################################
# 2/ NEIGHBOUR DEFAULTS AND VALIDATION
######################################


################################
# 2.1/ Resolve operator defaults
################################


def _resolve_grid_neighbours_for_interpolator(
    operator: Interpolator, transform: rio.transform.Affine
) -> GridNeighbours:
    """Use the requested grid window, or choose one from the interpolation method."""

    from geoutils.operators.interpolator import Kriging

    configured = operator.default_neighborhood
    if isinstance(configured, GridNeighbours):
        if configured.coverage != "center":
            raise ValueError("GridNeighbours coverage applies to Reducers, not Interpolators.")
        return configured
    if isinstance(configured, PointNeighbours):
        raise ValueError("PointNeighbours applies to point sources; use GridNeighbours for raster cells.")
    underlying = getattr(operator, "_wrapped_operator", operator)
    if isinstance(underlying, Kriging):
        return _build_kriging_grid_neighbours(underlying, transform)
    return GridNeighbours(size=3)


def _resolve_point_neighbours_for_interpolator(operator: Interpolator) -> PointNeighbours:
    """Use the requested point limits, or choose them from the interpolation method."""

    from geoutils.operators.interpolator import Kriging, Nearest

    operator = getattr(operator, "_wrapped_operator", operator)
    configured = operator.default_neighborhood
    if isinstance(configured, PointNeighbours):
        return configured
    if isinstance(configured, GridNeighbours):
        raise ValueError("GridNeighbours applies to raster cells; use PointNeighbours for point sources.")
    if isinstance(operator, Kriging):
        return PointNeighbours(radius=operator.max_overlap)
    if isinstance(operator, Nearest):
        return PointNeighbours(k=1)
    return PointNeighbours(k=8)


def _resolve_point_neighbours_for_reducer(operator: Reducer) -> PointNeighbours | None:
    """Use a reducer's point limits, or let grid() select points with its pixel radius."""

    configured = operator.default_neighborhood
    if isinstance(configured, GridNeighbours):
        raise ValueError("GridNeighbours applies to raster cells; use PointNeighbours for point sources.")
    return configured


###################################################
# 2.2/ Build kriging windows from a physical radius
###################################################


def _build_kriging_grid_neighbours(
    operator: Kriging, transform: Any, *, aligned_targets: bool = False
) -> GridNeighbours:
    """Find all raster cell offsets that may fall within the kriging radius of a target."""

    operator.validate_geospatial_support(2)
    basis = np.asarray([[transform.a, transform.b], [transform.d, transform.e]], dtype=np.float64)
    if not np.all(np.isfinite(basis)) or np.isclose(np.linalg.det(basis), 0):
        raise ValueError("Raster transform must define an invertible finite basis for kriging.")

    # Convert the search radius from map units to rows/columns (the grid may be rotated or skewed)
    # We add half a cell because the target can lie anywhere inside it; kriging checks the exact distances later
    inverse_basis = np.linalg.inv(basis)
    column_radius = int(np.ceil(np.linalg.norm(inverse_basis[0]) * operator.max_overlap + 0.5))
    row_radius = int(np.ceil(np.linalg.norm(inverse_basis[1]) * operator.max_overlap + 0.5))
    offsets = []
    for row in range(-row_radius, row_radius + 1):
        for column in range(-column_radius, column_radius + 1):
            if aligned_targets:
                # When targets lie at source cell centers, we can already exclude cells beyond the radius
                delta_x = transform.a * column + transform.b * row
                delta_y = transform.d * column + transform.e * row
                if np.hypot(delta_x, delta_y) > operator.max_overlap * (1 + 1e-12):
                    continue
            offsets.append((row, column))
    if not offsets:
        offsets.append((0, 0))
    return GridNeighbours(tuple(offsets))


####################################################
# 2.3/ Check interpolation windows and chunk overlap
####################################################


def _check_regular_grid_neighbours(
    operator: Interpolator, method: ScipyInterpolationMethod, *, warn: bool = False
) -> None:
    """Check that a requested raster window contains the method's natural stencil.

    Nearest uses an offset from its nearest cell. Linear and PCHIP use offsets from the lower grid index on each
    axis, so their stencils follow the target's position between cells rather than a centered odd-sized window.
    Larger windows do not change these calculations. Fitted cubic/quintic splines have no fixed local stencil.
    """

    neighborhood = operator.default_neighborhood
    if neighborhood is None:
        return
    if not isinstance(neighborhood, GridNeighbours):
        raise ValueError("PointNeighbours applies to point sources; use GridNeighbours for raster cells.")
    if neighborhood.coverage != "center":
        raise ValueError("GridNeighbours coverage applies to Reducers, not Interpolators.")
    if method in ("cubic", "quintic", "splinef2d"):
        raise ValueError(
            f"Raster {method} interpolation fits a spline over the grid and does not support GridNeighbours."
        )

    # These offsets cover every target position within one grid interval; boundary handling remains with SciPy
    if method == "nearest":
        required = {(0, 0)}
    else:
        axis_offsets = (-1, 0, 1, 2) if method == "pchip" else (0, 1)
        required = {(row, col) for row in axis_offsets for col in axis_offsets}
    supplied = set(neighborhood.offsets)
    if not required.issubset(supplied):
        raise ValueError(
            f"GridNeighbours does not contain the natural {method} interpolation stencil: {sorted(required)}."
        )
    if warn and supplied != required:
        warnings.warn(
            f"Raster {method} interpolation uses its natural stencil; extra GridNeighbours offsets are ignored.",
            UserWarning,
            stacklevel=3,
        )


def _compute_resampling_overlap(method: ScipyInterpolationMethod | Interpolator | Reducer) -> int:
    """Return the extra source cells that a chunk needs along each edge."""

    from geoutils.operators.interpolator import (
        INTERPOLATION_ORDERS,
        _regular_interpolation_method,
        _resolve_interpolator,
    )
    from geoutils.operators.reducer import Reducer

    operator = method if isinstance(method, Reducer) else _resolve_interpolator(method)
    method_name = None if isinstance(operator, Reducer) else _regular_interpolation_method(operator)
    if method_name is not None:
        return INTERPOLATION_ORDERS[method_name] + 1
    if isinstance(operator.default_neighborhood, GridNeighbours):
        return max(operator.default_neighborhood.overlap) + 1
    raise ValueError("Chunked raster resampling requires a built-in method or a GridNeighbours neighborhood.")


##########################
# 3/ RASTER CELL SELECTION
##########################


#####################################
# 3.1/ Collect cells at fixed offsets
#####################################


def _grid_window_kernel(
    neighborhood: GridNeighbours,
    *,
    phase: tuple[float, float] = (0.5, 0.5),
    transform: rio.transform.Affine | None = None,
    max_cells: int | None = None,
) -> NDArrayNum | None:
    """Calculate cell weights for a window, centered at the supplied fractional row/column position.

    The middle element represents the target's containing cell. Positive offsets select cells below/right of it.
    Covered windows use the same square or circular intersections as resample_at_points(). Return None when
    the dense kernel would exceed max_cells, so widely separated offsets can be evaluated directly.
    """

    offsets = np.asarray(neighborhood.offsets)
    depth = np.max(np.abs(offsets), axis=0)
    if neighborhood.coverage == "center":
        if max_cells is not None and np.prod(2 * depth + 1) > max_cells:
            return None
        kernel = np.zeros(tuple(2 * depth + 1), dtype=float)
        kernel[offsets[:, 0] + depth[0], offsets[:, 1] + depth[1]] = 1
        return kernel

    # Area coverage needs a geometric window, rather than an arbitrary set of offsets
    size = int(2 * np.max(depth) + 1)
    if len(offsets) < size:
        raise ValueError("Area filtering requires a square or circular GridNeighbours window.")
    shape = neighborhood.window_shape
    if shape is None:
        shapes: tuple[Literal["square", "circular"], ...] = ("square", "circular")
        for candidate in shapes:
            if set(neighborhood.offsets) == set(GridNeighbours(size=size, shape=candidate).offsets):
                shape = candidate
                break
    if shape is None or set(neighborhood.offsets) != set(GridNeighbours(size=size, shape=shape).offsets):
        raise ValueError("Area filtering requires a square or circular GridNeighbours window.")
    if max_cells is not None and (size + 2) ** 2 > max_cells:
        return None

    # Measure one footprint with enough surrounding cells to include every partly covered edge
    from affine import Affine

    padding = size // 2 + 1
    center_row, center_col = padding + phase[0], padding + phase[1]
    array_shape = (2 * padding + 1, 2 * padding + 1)
    if transform is None:
        transform = Affine(1, 0, 0, 0, -1, array_shape[0])
    if shape == "circular":
        angles = np.linspace(0, 2 * np.pi, 512, endpoint=False)
        radius = size / 2
        corners = np.column_stack((center_col + radius * np.cos(angles), center_row + radius * np.sin(angles)))
        x, y = _affine_matmul(transform, (corners[:, 0], corners[:, 1]))
        overlap = _grid_intersection_fractions(shapely.polygons(np.column_stack((x, y))[None]), transform, array_shape)
    else:
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * size / 2
        corners += (center_col, center_row)
        x, y = _affine_matmul(transform, (corners[:, 0], corners[:, 1]))
        overlap = _grid_intersection_fractions_from_corners(np.column_stack((x, y))[None], transform, array_shape)
    kernel = np.zeros(array_shape, dtype=float)
    kernel[overlap.rows, overlap.columns] = overlap.fractions
    if neighborhood.coverage == "all_touched":
        kernel = (kernel > 0).astype(float)

    # Remove unused outer rows/columns while preserving the containing cell at the center
    rows, cols = np.nonzero(kernel)
    row_depth = int(np.max(np.abs(rows - padding)))
    col_depth = int(np.max(np.abs(cols - padding)))
    return kernel[padding - row_depth : padding + row_depth + 1, padding - col_depth : padding + col_depth + 1]


def _create_circular_mask(
    shape: tuple[int, int], center: tuple[int, int] | None = None, radius: float | None = None
) -> NDArrayBool:
    """Create a circular kernel, using the array centre and half width by default."""

    # Use the array center and its nearest edge to choose a fully contained default circle
    w, h = shape

    if center is None:
        center = (int(w / 2), int(h / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])

    # Select cells strictly inside the radius to preserve the patch kernel boundary convention
    Y, X = np.ogrid[:w, :h]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center < radius

    return mask


def _prepare_grid_neighbours_data(
    array: NDArrayNum,
    transform: rio.transform.Affine,
    points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum],
    operator: Interpolator | Reducer,
    *,
    area_or_point: Literal["Area", "Point"] | None,
    shift_area_or_point: bool | None,
    nodata_propagation: NodataPropagation,
    neighborhood: GridNeighbours | None = None,
    band: int = 1,
    source_index_offset: tuple[int, int] = (0, 0),
    source_shape: tuple[int, int] | None = None,
) -> tuple[list[LocalData], NodataHandling | None, NDArrayNum, NDArrayNum]:
    """Collect raster cells at fixed offsets around each target point for an interpolator or reducer."""

    from geoutils.operators.interpolator import Interpolator
    from geoutils.operators.reducer import Reducer
    from geoutils.raster.referencing import _xy2ij

    if neighborhood is not None:
        neighbours = neighborhood
    elif isinstance(operator, Interpolator):
        neighbours = _resolve_grid_neighbours_for_interpolator(operator, transform)
    else:
        neighbours = GridNeighbours(size=1)
    x = np.atleast_1d(np.asarray(points[0]))
    y = np.atleast_1d(np.asarray(points[1]))
    source_rows, source_cols = _xy2ij(
        x,
        y,
        transform=transform,
        area_or_point=area_or_point,
        shift_area_or_point=shift_area_or_point,
    )

    # Apply the window offsets around the source cell containing each requested point
    center_rows = np.floor(source_rows).astype(np.int64)
    center_cols = np.floor(source_cols).astype(np.int64)
    global_shape = (array.shape[0], array.shape[1]) if source_shape is None else source_shape
    band_offset = (band - 1) * global_shape[0] * global_shape[1]
    is_reducer = isinstance(operator, Reducer)
    values_data = np.asanyarray(np.ma.getdata(array))
    values_mask = np.ma.getmaskarray(array) if np.ma.isMaskedArray(array) else None
    local_inputs = []
    handling: NodataHandling | None
    if nodata_propagation == "nearest":
        handling = "ignore"
    else:
        handling = nodata_propagation

    for target_index, (center_row, center_col) in enumerate(zip(center_rows, center_cols)):
        # We need indexes in both the current array (which may be a chunk) and the complete raster
        # Exclude cells outside either one before reading their values
        local_rows = np.asarray([center_row + offset[0] for offset in neighbours.offsets], dtype=np.int64)
        local_cols = np.asarray([center_col + offset[1] for offset in neighbours.offsets], dtype=np.int64)
        global_rows = local_rows + source_index_offset[0]
        global_cols = local_cols + source_index_offset[1]
        inside = (local_rows >= 0) & (local_rows < array.shape[0])
        inside &= (local_cols >= 0) & (local_cols < array.shape[1])
        inside &= (global_rows >= 0) & (global_rows < global_shape[0])
        inside &= (global_cols >= 0) & (global_cols < global_shape[1])
        if is_reducer and (
            center_row + source_index_offset[0] < 0
            or center_row + source_index_offset[0] >= global_shape[0]
            or center_col + source_index_offset[1] < 0
            or center_col + source_index_offset[1] >= global_shape[1]
        ):
            # A reduction needs a source cell containing the target, even when its window reaches the raster
            inside[:] = False
        local_rows, local_cols = local_rows[inside], local_cols[inside]
        global_rows, global_cols = global_rows[inside], global_cols[inside]

        # Use the same cell order for values + coordinates, with IDs based on the complete raster
        values = np.asarray(values_data[local_rows, local_cols])
        valid = np.isfinite(values)
        if values_mask is not None:
            valid &= ~values_mask[local_rows, local_cols]
        source_ids = band_offset + global_rows * global_shape[1] + global_cols

        # Distances are measured from cell centers to the requested X/Y position
        source_x = transform.a * (local_cols + 0.5) + transform.b * (local_rows + 0.5) + transform.c
        source_y = transform.d * (local_cols + 0.5) + transform.e * (local_rows + 0.5) + transform.f
        coordinates = np.column_stack((source_x, source_y))
        target = np.asarray([x[target_index], y[target_index]])
        local_inputs.append(
            LocalData(
                values=values,
                valid=valid,
                source_ids=source_ids,
                coordinates=coordinates,
                target=target,
                distances=np.linalg.norm(coordinates - target, axis=1),
            )
        )
    return local_inputs, handling, source_rows, source_cols


#########################################
# 3.2/ Measure fractional window coverage
#########################################


def _prepare_fractional_window_data(
    array: NDArrayNum,
    transform: rio.transform.Affine,
    points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum],
    *,
    area_or_point: Literal["Area", "Point"] | None,
    fractional_window: int,
    fractional_shape: Literal["square", "circular"] | None,
    coverage: GridCoverage = "fractional",
    band: int,
    source_index_offset: tuple[int, int] = (0, 0),
    source_shape: tuple[int, int] | None = None,
) -> tuple[list[LocalData], NDArrayNum, NDArrayNum]:
    """
    Select cells touched by a window centered on each point, with optional area weights.

    It also returns each point's row and column position in the source raster so _resample_at_points() can check
    the corresponding cell against the nodata mask.
    """

    from geoutils.raster.referencing import _xy2ij

    # Floating row and column positions center each footprint on the requested point
    source_rows, source_cols = _xy2ij(
        points[0], points[1], transform, area_or_point, op=np.float64, shift_area_or_point=False
    )
    if fractional_shape == "circular":
        # Match the circle used to select cell centers for this window size
        radius = fractional_window / 2
        # Approximate the curved boundary before measuring how much of each cell it covers
        angles = np.linspace(0, 2 * np.pi, 512, endpoint=False)
        footprint_rows = source_rows[:, None] + radius * np.sin(angles)
        footprint_cols = source_cols[:, None] + radius * np.cos(angles)
    else:
        # A square needs only its four corners, including cells partly covered at the edges
        half_size = fractional_window / 2
        footprint_rows = source_rows[:, None] + np.array([-half_size, -half_size, half_size, half_size])
        footprint_cols = source_cols[:, None] + np.array([-half_size, half_size, half_size, -half_size])

    # Map each footprint from raster positions into the coordinates used by the overlap calculation
    footprint_x = transform.a * footprint_cols + transform.b * footprint_rows + transform.c
    footprint_y = transform.d * footprint_cols + transform.e * footprint_rows + transform.f
    footprint = np.stack((footprint_x, footprint_y), axis=-1)

    # Measure covered cell fractions and attach them to the values passed to the Reducer
    shape = (array.shape[0], array.shape[1])
    if fractional_shape == "circular":
        overlap = _grid_intersection_fractions(shapely.polygons(footprint), transform, shape)
    else:
        overlap = _grid_intersection_fractions_from_corners(footprint, transform, shape)
    targets = np.column_stack(points)
    global_shape = (array.shape[0], array.shape[1]) if source_shape is None else source_shape
    local_inputs = _grid_intersection_local_data(
        array,
        transform,
        overlap,
        source_id_offset=(band - 1) * global_shape[0] * global_shape[1],
        source_index_offset=source_index_offset,
        source_shape=global_shape,
        targets=targets,
    )
    if coverage == "all_touched":
        local_inputs = [replace(local, support_weights=None) for local in local_inputs]

    # Exclude padding beyond the complete raster, including partly covered edge cells
    for index, local in enumerate(local_inputs):
        rows, columns, _ = overlap.for_geometry(index)
        global_rows = rows + source_index_offset[0]
        global_columns = columns + source_index_offset[1]
        inside_cells = (global_rows >= 0) & (global_rows < global_shape[0])
        inside_cells &= (global_columns >= 0) & (global_columns < global_shape[1])
        local_inputs[index] = local.select(inside_cells)

    # As with an ordinary window, a point outside the raster does not produce a reduction
    global_rows = source_rows + source_index_offset[0]
    global_cols = source_cols + source_index_offset[1]
    inside = (global_rows >= 0) & (global_rows < global_shape[0])
    inside &= (global_cols >= 0) & (global_cols < global_shape[1])
    local_inputs = [
        local if inside[index] else local.select(np.zeros(len(local.values), dtype=bool))
        for index, local in enumerate(local_inputs)
    ]
    return local_inputs, source_rows, source_cols


#######################################################
# 3.3/ Prepare regular interpolation values and weights
#######################################################


def _prepare_regular_interpolation_data(
    array: NDArrayNum,
    transform: rio.transform.Affine,
    points: tuple[Number, Number] | tuple[NDArrayNum, NDArrayNum],
    method: ScipyInterpolationMethod | Interpolator,
    *,
    area_or_point: Literal["Area", "Point"] | None,
    shift_area_or_point: bool | None,
    nodata_propagation: NodataPropagation,
    band: int = 1,
    array_indices: tuple[NDArrayNum, NDArrayNum] | None = None,
    source_index_offset: tuple[int, int] = (0, 0),
    source_shape: tuple[int, int] | None = None,
    index_type: type[np.float32] | type[np.float64] = np.float32,
) -> tuple[Interpolator, list[LocalData], NodataHandling | None]:
    """Collect the cells and weights needed to propagate uncertainty through raster interpolation.

    Nearest/linear interpolation have known weights, which we calculate directly. Other regular-grid methods repeat
    the original raster interpolation. Custom methods receive the cells selected by _prepare_grid_neighbours_data().
    """

    from geoutils.operators.interpolator import (
        _regular_interpolation_method,
        _RegularGridPrediction,
        _RegularGridWeights,
        _resolve_interpolator,
    )
    from geoutils.raster.referencing import _xy2ij

    operator = _resolve_interpolator(method)
    method_name = _regular_interpolation_method(operator)
    if method_name is None:
        grid_inputs, handling, _, _ = _prepare_grid_neighbours_data(
            array,
            transform,
            points,
            operator,
            area_or_point=area_or_point,
            shift_area_or_point=shift_area_or_point,
            nodata_propagation=nodata_propagation,
            band=band,
            source_index_offset=source_index_offset,
            source_shape=source_shape,
        )
        return operator, grid_inputs, handling

    if method_name not in {"nearest", "linear"}:
        # Fitted splines can depend on distant source values; preserve the complete raster and its missing cells
        source = np.asarray(np.ma.getdata(array))
        valid = np.isfinite(source) & ~np.ma.getmaskarray(array)
        rows, cols = np.nonzero(valid)
        global_shape = source.shape if source_shape is None else source_shape
        source_ids = (
            (band - 1) * np.prod(global_shape)
            + (rows + source_index_offset[0]) * global_shape[1]
            + cols
            + source_index_offset[1]
        )
        coordinates = np.column_stack(
            (
                transform.a * (cols + 0.5) + transform.b * (rows + 0.5) + transform.c,
                transform.d * (cols + 0.5) + transform.e * (rows + 0.5) + transform.f,
            )
        )
        values = source[rows, cols]
        spline_inputs = [
            LocalData(
                values=values,
                valid=np.ones(len(values), dtype=bool),
                source_ids=source_ids,
                coordinates=coordinates,
                target=np.asarray([x, y]),
            )
            for x, y in zip(np.atleast_1d(points[0]), np.atleast_1d(points[1]))
        ]
        prediction = _RegularGridPrediction(
            (source.shape[0], source.shape[1]),
            source.dtype,
            transform,
            method_name,
            area_or_point,
            shift_area_or_point,
            nodata_propagation,
        )
        return prediction, spline_inputs, "ignore"

    # Convert X/Y coordinates to array indexes (including fractions between cells)
    x = np.atleast_1d(np.asarray(points[0]))
    y = np.atleast_1d(np.asarray(points[1]))
    if array_indices is None:
        # Match direct interpolation's pixel index precision
        source_rows, source_cols = _xy2ij(
            x,
            y,
            transform=transform,
            area_or_point=area_or_point,
            shift_area_or_point=shift_area_or_point,
            op=index_type,
        )
    else:
        source_rows, source_cols = array_indices
    source = np.asarray(np.ma.getdata(array))
    valid = np.isfinite(source)
    if np.ma.isMaskedArray(array):
        valid &= ~np.ma.getmaskarray(array)
    has_missing = not np.all(valid)
    global_shape = source.shape if source_shape is None else source_shape
    source_id_offset = (band - 1) * global_shape[0] * global_shape[1]
    weighted_operator = _RegularGridWeights()
    local_inputs: list[LocalData] = []

    for target_x, target_y, source_row, source_col in zip(x, y, source_rows, source_cols):
        # Nearest and linear interpolation accept targets up to half a pixel beyond the outer cell centers
        inside = -0.5 <= source_row < source.shape[0] - 0.5
        inside &= -0.5 <= source_col < source.shape[1] - 0.5
        if not inside:
            local_inputs.append(
                LocalData(
                    values=np.empty(0, dtype=float),
                    valid=np.empty(0, dtype=bool),
                    source_ids=np.empty(0, dtype=np.int64),
                    coordinates=np.empty((0, 2), dtype=float),
                    target=np.asarray([target_x, target_y]),
                    distances=np.empty(0, dtype=float),
                    interpolation_weights=np.empty(0, dtype=float),
                )
            )
            continue

        # Use the nearest cell or the four surrounding cells for linear interpolation
        # Clipping the indexes follows SciPy's behavior at the array boundary
        clipped_row = float(np.clip(source_row, 0, source.shape[0] - 1))
        clipped_col = float(np.clip(source_col, 0, source.shape[1] - 1))
        if method_name == "nearest":
            candidates = [(int(np.floor(clipped_row + 0.5)), int(np.floor(clipped_col + 0.5)), 1.0)]
        else:
            lower_row = int(np.floor(clipped_row))
            lower_col = int(np.floor(clipped_col))
            upper_row = min(lower_row + 1, source.shape[0] - 1)
            upper_col = min(lower_col + 1, source.shape[1] - 1)
            row_fraction = clipped_row - lower_row
            col_fraction = clipped_col - lower_col
            candidates = [
                (lower_row, lower_col, (1 - row_fraction) * (1 - col_fraction)),
                (lower_row, upper_col, (1 - row_fraction) * col_fraction),
                (upper_row, lower_col, row_fraction * (1 - col_fraction)),
                (upper_row, upper_col, row_fraction * col_fraction),
            ]

        # At the array edge, some of the four cells coincide, so we add their weights together
        # Exclude invalid values and rescale the remaining weights to sum to one
        combined: dict[tuple[int, int], float] = {}
        for row, col, weight in candidates:
            if weight > 0 and valid[row, col]:
                combined[(row, col)] = combined.get((row, col), 0.0) + weight
        total = sum(combined.values())
        if has_missing:
            # Match the float32 validity interpolation used to normalize the original raster calculation
            total = float(np.float32(total))
        selected = list(combined)
        rows = np.asarray([row for row, _ in selected], dtype=np.int64)
        cols = np.asarray([col for _, col in selected], dtype=np.int64)
        weights = np.asarray([combined[index] / total for index in selected], dtype=float) if total > 0 else np.empty(0)

        # Record cell IDs + coordinates so the uncertainty calculation can identify shared source values
        source_x = transform.a * (cols + 0.5) + transform.b * (rows + 0.5) + transform.c
        source_y = transform.d * (cols + 0.5) + transform.e * (rows + 0.5) + transform.f
        coordinates = np.column_stack((source_x, source_y))
        target = np.asarray([target_x, target_y])
        local_inputs.append(
            LocalData(
                values=source[rows, cols],
                valid=np.ones(len(rows), dtype=bool),
                source_ids=source_id_offset
                + (rows + source_index_offset[0]) * global_shape[1]
                + cols
                + source_index_offset[1],
                coordinates=coordinates,
                target=target,
                distances=np.linalg.norm(coordinates - target, axis=1),
                interpolation_weights=weights,
            )
        )
    return weighted_operator, local_inputs, "ignore"


##########################
# 4/ POINT CLOUD SELECTION
##########################


_GRID_QUERY_ROWS = 128


######################################################
# 4.1/ Read coordinates, values and source identifiers
######################################################


def _prepare_point_gridding_data(
    pc: gpd.GeoDataFrame,
    data_column_name: str | None,
) -> tuple[NDArrayNum, NDArrayNum, NDArrayNum, NDArrayBool]:
    """
    Read point coordinates and values, separating finite values from all points with usable X/Y coordinates.

    :param pc: Input point cloud.
    :param data_column_name: Name of the data column, or None to use geometry elevations.

    :returns: Coordinates and values for valid points, then coordinates and validity flags for all points with
        finite X/Y coordinates (including points with nodata values).
    """

    values = np.asarray(pc[data_column_name].values if data_column_name is not None else pc.geometry.z.values)
    points = np.column_stack((pc.geometry.x.values, pc.geometry.y.values))

    # Coordinates must be finite for either values or nodata positions to affect the output
    finite_coordinates = np.isfinite(points).all(axis=1)
    valid = finite_coordinates & np.isfinite(values)
    valid_points = np.ascontiguousarray(points[valid], dtype=np.float64)
    valid_values = np.ascontiguousarray(values[valid], dtype=np.float64)

    # If all points with finite X/Y coordinates have valid values, reuse their coordinate array
    if np.array_equal(valid, finite_coordinates):
        positioned_points = valid_points
        positioned_valid = np.ones(len(valid_points), dtype=np.bool_)
    else:
        positioned_points = np.ascontiguousarray(points[finite_coordinates], dtype=np.float64)
        positioned_valid = np.ascontiguousarray(np.isfinite(values[finite_coordinates]), dtype=np.bool_)
    return (
        valid_points,
        valid_values,
        positioned_points,
        positioned_valid,
    )


def _prepare_point_operator_data(
    pc: gpd.GeoDataFrame,
    data_column_name: str | None,
) -> tuple[NDArrayNum, NDArrayNum, NDArrayBool, NDArray[Any]]:
    """Read point coordinates, values, validity flags and unique row IDs for an interpolator or reducer."""

    raw_values = np.asarray(pc[data_column_name].values if data_column_name is not None else pc.geometry.z.values)
    raw_points = np.column_stack((pc.geometry.x.values, pc.geometry.y.values))
    finite_coordinates = np.isfinite(raw_points).all(axis=1)

    # Include nodata values here: the interpolator/reducer decides whether to ignore or propagate them
    points = np.ascontiguousarray(raw_points[finite_coordinates], dtype=np.float64)
    values = np.ascontiguousarray(raw_values[finite_coordinates], dtype=np.float64)
    valid = np.isfinite(values)
    source_ids = np.asarray(pc.index)[finite_coordinates]

    # A dataframe index can contain duplicates; use row positions instead when it cannot identify points uniquely
    if not pd.Index(source_ids.tolist()).is_unique:
        source_ids = np.flatnonzero(finite_coordinates).astype(np.int64)
    return points, values, valid, source_ids


#############################
# 4.2/ Build spatial searches
#############################


def _compute_query_distance_upper_bound(radius: float | None) -> float:
    """Widen SciPy's strict query bound enough to find points exactly at the requested radius."""

    if radius is None:
        return np.inf
    # A subnormal bound squares to zero inside cKDTree, so a zero radius needs a representable squared bound
    return float(max(np.nextafter(radius, np.inf), np.sqrt(np.finfo(np.float64).tiny)))


def _build_grid_queries(x_coords: NDArrayNum, y_coords: NDArrayNum) -> NDArrayNum:
    """Return flattened coordinates for a bounded group of output rows."""

    # Filling two vectors avoids retaining complete X/Y meshgrids for a large raster
    queries = np.empty((len(x_coords) * len(y_coords), 2), dtype=np.float64)
    queries[:, 0] = np.tile(x_coords, len(y_coords))
    queries[:, 1] = np.repeat(y_coords, len(x_coords))
    return queries


def _build_scaled_point_tree(points: NDArrayNum, x_start: float, y_start: float, res_x: float, res_y: float) -> cKDTree:
    """Build a spatial tree whose distances are expressed in output pixels."""

    scaled_points = np.empty_like(points, dtype=np.float64)
    scaled_points[:, 0] = (points[:, 0] - x_start) / res_x
    scaled_points[:, 1] = (points[:, 1] - y_start) / res_y
    return cKDTree(scaled_points)


#########################################
# 4.3/ Collect neighbours for each target
#########################################


def _remove_self_pairs(
    target_indexes: NDArrayNum,
    source_indexes: NDArrayNum,
    source_coordinates: NDArrayNum,
    source_values: NDArrayNum,
    target_coordinates: NDArrayNum,
    target_values: NDArrayNum,
    source_ids: NDArrayNum | None,
    target_ids: NDArrayNum | None,
) -> NDArrayBool:
    """Return a mask selecting neighbours other than the point being filtered (remove its own row only once)."""

    if len(target_indexes) == 0:
        return np.empty(0, dtype=bool)

    # Unique row IDs distinguish points that have identical coordinates and values
    if source_ids is not None and target_ids is not None:
        self_candidates = source_ids[source_indexes] == target_ids[target_indexes]
    else:
        same_coordinates = np.all(source_coordinates[source_indexes] == target_coordinates[target_indexes], axis=1)
        neighbor_values = source_values[source_indexes]
        current_values = target_values[target_indexes]
        same_values = (neighbor_values == current_values) | (np.isnan(neighbor_values) & np.isnan(current_values))
        self_candidates = same_coordinates & same_values

    # Without row IDs, several identical points may match the target, so remove only the first match
    selected_source = np.full(len(target_coordinates), len(source_coordinates), dtype=np.int64)
    np.minimum.at(selected_source, target_indexes[self_candidates], source_indexes[self_candidates])
    remove = self_candidates & (source_indexes == selected_source[target_indexes])
    return ~remove


def _query_point_neighbours(
    source_tree: Any,
    source_coordinates: NDArrayNum,
    source_values: NDArrayNum,
    source_ids: NDArrayNum | None,
    target_coordinates: NDArrayNum,
    target_values: NDArrayNum,
    target_ids: NDArrayNum | None,
    neighborhood: PointNeighbours,
    include_self: bool,
    n_threads: int,
) -> tuple[NDArrayNum, NDArrayNum, NDArrayNum]:
    """Find each target's neighbours and return their target/source indexes + X/Y distances."""

    from scipy.spatial import cKDTree

    if neighborhood.k is not None:
        # Request one extra neighbor when we will remove the target row from the results
        query_count = neighborhood.k + int(not include_self)
        query_count = min(query_count, len(source_coordinates))
        distance_limit = _compute_query_distance_upper_bound(neighborhood.radius)
        distances, indexes = source_tree.query(
            target_coordinates,
            k=query_count,
            distance_upper_bound=distance_limit,
            workers=n_threads,
        )
        indexes = np.asarray(indexes)
        distances = np.asarray(distances)
        if indexes.ndim == 1:
            indexes = indexes[:, None]
            distances = distances[:, None]
        target_indexes = np.repeat(np.arange(len(target_coordinates)), indexes.shape[1])
        source_indexes = indexes.reshape(-1)
        distances = distances.reshape(-1)

        # SciPy marks missing neighbors with a source index one past the end of the source array
        present = source_indexes < len(source_coordinates)
        if neighborhood.radius is not None:
            present &= distances <= neighborhood.radius
        target_indexes = target_indexes[present]
        source_indexes = source_indexes[present]
        distances = distances[present]
    else:
        assert neighborhood.radius is not None
        # Store only point pairs within the radius, instead of a full target-by-source distance matrix
        target_tree = cKDTree(target_coordinates)
        neighbors = target_tree.sparse_distance_matrix(source_tree, neighborhood.radius, output_type="coo_matrix")
        target_indexes = np.asarray(neighbors.row, dtype=np.int64)
        source_indexes = np.asarray(neighbors.col, dtype=np.int64)
        distances = np.asarray(neighbors.data, dtype=np.float64)

    if not include_self:
        keep = _remove_self_pairs(
            target_indexes,
            source_indexes,
            source_coordinates,
            source_values,
            target_coordinates,
            target_values,
            source_ids,
            target_ids,
        )
        target_indexes = target_indexes[keep]
        source_indexes = source_indexes[keep]
        distances = distances[keep]

        # After removing each target's own row, select at most k neighbours from the remaining candidates
        if neighborhood.k is not None and len(target_indexes) > 0:
            # Sort neighbors by target and distance, then select the first k neighbors for each target
            order = np.lexsort((source_indexes, distances, target_indexes))
            ordered_targets = target_indexes[order]
            group_starts = np.flatnonzero(np.r_[True, ordered_targets[1:] != ordered_targets[:-1]])
            group_lengths = np.diff(np.r_[group_starts, len(order)])
            rank = np.arange(len(order)) - np.repeat(group_starts, group_lengths)
            keep_ordered = rank < neighborhood.k
            order = order[keep_ordered]
            target_indexes = target_indexes[order]
            source_indexes = source_indexes[order]
            distances = distances[order]
    return target_indexes, source_indexes, distances


def _prepare_point_neighbours_data(
    pc: gpd.GeoDataFrame,
    grid_coords: tuple[NDArrayNum, NDArrayNum],
    data_column_name: str | None,
    operator: Interpolator | Reducer,
    *,
    res_x: float,
    res_y: float,
    radius: float,
    min_points: int,
) -> tuple[list[LocalData], list[tuple[int, int]]]:
    """Find the points used by an interpolator/reducer for each output grid cell."""

    from geoutils.operators.execution import _get_builtin_gridding_method
    from geoutils.operators.interpolator import Interpolator

    points, values, valid, source_ids = _prepare_point_operator_data(pc, data_column_name=data_column_name)
    if len(points) == 0:
        return [], []
    builtin_method = _get_builtin_gridding_method(operator, default_neighborhood_only=False)
    default_geometry = operator.default_neighborhood is None and builtin_method in ("nearest", "linear", "cubic")
    finite_indexes = np.flatnonzero(valid)
    finite_tree = cKDTree(points[finite_indexes] / [res_x, res_y]) if default_geometry and len(finite_indexes) else None
    nearest_tree = cKDTree(points[finite_indexes]) if default_geometry and len(finite_indexes) else None

    # Configured point neighborhoods use coordinate units; default reducers use grid()'s radius in output pixels
    neighborhood: PointNeighbours | None
    if isinstance(operator, Interpolator):
        neighborhood = _resolve_point_neighbours_for_interpolator(operator)
        if builtin_method == "idw" and operator.default_neighborhood is None:
            neighborhood = None
    else:
        neighborhood = _resolve_point_neighbours_for_reducer(operator)
    if neighborhood is not None:
        point_tree = cKDTree(points)
    else:
        if not np.isfinite(radius):
            raise ValueError("Gridding Reducers require a finite dist_nodata_pixel support radius.")
        x_start = float(np.min(grid_coords[0]))
        y_start = float(np.min(grid_coords[1]))
        point_tree = _build_scaled_point_tree(points, x_start=x_start, y_start=y_start, res_x=res_x, res_y=res_y)

    # Collect each cell's neighbours before calculating the output values
    local_inputs = []
    output_indexes = []
    indexes: NDArrayNum
    for row, target_y in enumerate(grid_coords[1]):
        if neighborhood is not None and not default_geometry:
            # Query a row together, sharing the point-filter search and its radius/count rules
            targets = np.column_stack((grid_coords[0], np.full(len(grid_coords[0]), target_y)))
            target_ids, sources, neighbor_distances = _query_point_neighbours(
                point_tree,
                points,
                values,
                source_ids,
                targets,
                np.empty(len(targets)),
                None,
                neighborhood,
                True,
                1,
            )
            order = np.lexsort((sources, neighbor_distances, target_ids))
            sources, neighbor_distances = sources[order], neighbor_distances[order]
            starts = np.searchsorted(target_ids[order], np.arange(len(targets) + 1))
        for col, target_x in enumerate(grid_coords[0]):
            target = np.asarray([target_x, target_y], dtype=np.float64)
            if default_geometry:
                # Built-in triangulation uses all finite points; nearest uses the closest finite observation
                if finite_tree is None:
                    continue
                nearest_distance, _ = finite_tree.query(target / [res_x, res_y])
                if nearest_distance > radius:
                    continue
                assert nearest_tree is not None
                _, nearest_index = nearest_tree.query(target)
                indexes = finite_indexes[[nearest_index]] if builtin_method == "nearest" else finite_indexes
                distances = np.linalg.norm(points[indexes] - target, axis=1)
            elif neighborhood is not None:
                indexes = sources[starts[col] : starts[col + 1]]
                distances = neighbor_distances[starts[col] : starts[col + 1]]
            else:
                scaled_target = np.asarray([(target_x - x_start) / res_x, (target_y - y_start) / res_y])
                indexes = np.asarray(point_tree.query_ball_point(scaled_target, radius), dtype=np.int64)
                distances = np.linalg.norm(points[indexes] - target, axis=1)

            # Exact IDW observations take precedence over min_points, just as in the optimized radius kernels
            exact_idw = builtin_method == "idw" and np.any(valid[indexes] & (distances == 0))
            if not default_geometry and np.count_nonzero(valid[indexes]) < min_points and not exact_idw:
                continue
            if builtin_method in ("linear", "cubic"):
                finite_points = points[indexes[valid[indexes]]]
                if len(finite_points) < 3 or np.linalg.matrix_rank(finite_points - finite_points[0]) < 2:
                    continue
            output_indexes.append((row, col))
            local_inputs.append(
                LocalData(
                    values=values[indexes],
                    valid=valid[indexes],
                    source_ids=source_ids[indexes],
                    coordinates=points[indexes],
                    target=target,
                    distances=distances,
                )
            )
    return local_inputs, output_indexes


######################################################
# 4.4/ Grid within a radius and mask unsupported cells
######################################################


def _grid_radius_scipy(
    points: NDArrayNum,
    values: NDArrayNum,
    grid_coords: tuple[NDArrayNum, NDArrayNum],
    res_x: float,
    res_y: float,
    radius: float,
    min_points: int,
    minimum_inputs: int,
    calculate: Callable[[NDArrayNum, Any, NDArrayNum, cKDTree, NDArrayNum, NDArrayBool], None],
) -> NDArrayNum:
    """Compute radius-based gridding with bounded SciPy sparse distance matrices."""

    x_coords, y_coords = grid_coords
    x_start = float(np.min(x_coords))
    y_start = float(np.min(y_coords))
    # Index source points in output-pixel coordinates so radius has the same scale on both axes
    point_tree = _build_scaled_point_tree(points, x_start=x_start, y_start=y_start, res_x=res_x, res_y=res_y)
    scaled_x = (x_coords - x_start) / res_x
    scaled_y = (y_coords - y_start) / res_y
    output = np.full((len(y_coords), len(x_coords)), np.nan, dtype=np.float64)

    # Process bounded row blocks instead of constructing all possible point-cell pairs at once
    for row_start in range(0, len(y_coords), _GRID_QUERY_ROWS):
        row_stop = min(row_start + _GRID_QUERY_ROWS, len(y_coords))
        queries = _build_grid_queries(scaled_x, scaled_y[row_start:row_stop])
        # Build a tree for this block of grid cells and retain only neighbors within radius
        query_tree = cKDTree(queries)
        pairs = query_tree.sparse_distance_matrix(point_tree, radius, output_type="coo_matrix")
        # Sparse-matrix rows identify grid cells, columns identify source points and data stores distances
        block = np.full(len(queries), np.nan, dtype=np.float64)
        counts = np.bincount(pairs.row, minlength=len(queries))
        required_points = max(minimum_inputs, min_points)
        valid = counts >= required_points

        calculate(block, pairs, queries, point_tree, counts, valid)
        output[row_start:row_stop] = block.reshape(row_stop - row_start, len(x_coords))
    return output


def _mask_grid_beyond_support(
    array: NDArrayNum,
    points: NDArrayNum,
    grid_coords: tuple[NDArrayNum, NDArrayNum],
    res_x: float,
    res_y: float,
    radius: float,
    n_threads: int,
) -> None:
    """Set cells farther than the support radius from every source point to NaN."""

    x_coords, y_coords = grid_coords
    x_start = float(np.min(x_coords))
    y_start = float(np.min(y_coords))
    # Index source points once in the output-pixel coordinate system
    point_tree = _build_scaled_point_tree(points, x_start=x_start, y_start=y_start, res_x=res_x, res_y=res_y)
    scaled_x = (x_coords - x_start) / res_x
    scaled_y = (y_coords - y_start) / res_y

    # Query only a bounded number of rows so the distance check remains memory efficient
    for row_start in range(0, len(y_coords), _GRID_QUERY_ROWS):
        row_stop = min(row_start + _GRID_QUERY_ROWS, len(y_coords))
        queries = _build_grid_queries(scaled_x, scaled_y[row_start:row_stop])
        # The nearest distance is sufficient to decide whether any point supports each cell
        distances, _ = point_tree.query(queries, k=1, workers=n_threads)
        block = array[row_start:row_stop]
        block[distances.reshape(block.shape) > radius] = np.nan
