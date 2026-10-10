"""Test how operators describe source-cell and point neighborhoods."""

from typing import Literal

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio
from affine import Affine

import geoutils as gu
from geoutils.operators import GridNeighbours, Interpolator, PointNeighbours, Reducer
from geoutils.operators.interpolator import Kriging, Nearest, ScipyInterpolationMethod
from geoutils.operators.neighbours import (
    _build_grid_queries,
    _build_kriging_grid_neighbours,
    _build_scaled_point_tree,
    _compute_resampling_overlap,
    _prepare_fractional_window_data,
    _prepare_grid_neighbours_data,
    _prepare_point_gridding_data,
    _prepare_point_neighbours_data,
    _prepare_point_operator_data,
    _prepare_regular_interpolation_data,
    _resolve_point_neighbours_for_interpolator,
)
from geoutils.operators.reducer import Mean
from geoutils.stats.variography import VariogramModel
from tests.operator_helpers import LocalMeanInterpolator

##########################
# 1/ NEIGHBOUR DEFINITIONS
##########################


class TestPointNeighbours:
    """Test module for point neighborhoods."""

    @pytest.mark.parametrize("k,radius", [(3, None), (None, 5.0), (3, 5.0)])
    def test_point_neighbours(self, k: int | None, radius: float | None) -> None:
        """Checks point neighborhood basic definition."""

        # Should accept max number of neighbors, radius or both
        neighborhood = PointNeighbours(k=k, radius=radius)
        assert neighborhood.k == k
        assert neighborhood.radius == radius

    def test_point_neighbours__zero_radius(self) -> None:
        """Checks edge case of zero radius works in principle (to select only center observation itself)."""

        neighborhood = PointNeighbours(k=2, radius=0)
        assert neighborhood.k == 2
        assert neighborhood.radius == 0

    def test_point_neighbours__error_missing(self) -> None:
        """Checks that a point neighborhood raises an error for missing max number/radius input."""

        with pytest.raises(ValueError, match="at least one"):
            PointNeighbours()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"k": 0},
            {"k": -1},
            {"k": True},
            {"k": 1.5},
            {"radius": -1},
            {"radius": float("inf")},
            {"radius": float("nan")},
            {"radius": True},
        ],
    )
    def test_point_neighbours__error_invalid(self, kwargs: dict[str, float]) -> None:
        """Checks that max number/radius input raise error for non-finite, negative, boolean or fractional values."""

        # Raise error on invalid input
        with pytest.raises(ValueError):
            PointNeighbours(**kwargs)  # type: ignore[arg-type]


class TestGridNeighbours:
    """Test module for raster neighborhoods."""

    def test_grid_neighbours__coverage(self) -> None:
        """Checks that a grid window records its cell coverage rule and rejects unknown rules."""

        # The same offsets can select complete cells or weight their covered area
        center = GridNeighbours(size=3)
        fractional = GridNeighbours(size=3, coverage="fractional")
        assert center.offsets == fractional.offsets
        assert center.coverage == "center"
        assert fractional.coverage == "fractional"

        # A misspelled rule cannot silently use center coverage
        with pytest.raises(ValueError, match="GridNeighbours coverage"):
            GridNeighbours(size=3, coverage="invalid")  # type: ignore

    @pytest.mark.parametrize("shape,expected_count", [("square", 25), ("circular", 21)])
    def test_grid_neighbours__size_constructor(self, shape: Literal["square", "circular"], expected_count: int) -> None:
        """Checks that the constructor selects centered square and circular windows of the requested size."""

        # A five-cell window reaches two cells from the center on each axis
        neighborhood = GridNeighbours(size=5, shape=shape)

        # Check cell count, center and extent for both shapes
        assert len(neighborhood.offsets) == expected_count
        assert (0, 0) in neighborhood.offsets
        assert neighborhood.window_shape == shape
        assert neighborhood.overlap == (2, 2)

        # A radius of 2.5 pixels includes cells one row from the edge, but excludes the corners
        if shape == "circular":
            assert (2, 1) in neighborhood.offsets
            assert (2, 2) not in neighborhood.offsets

        # Positional size creates the same square window
        if shape == "square":
            assert GridNeighbours(5) == neighborhood

    def test_gridneighb__default_window(self) -> None:
        """Checks square window creates proper offsets and overlap (for chunking)."""

        neighbours = GridNeighbours(size=3)

        # Offset should be centered and "row-major"
        assert neighbours.offsets == (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 0),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )

        # A 3x3 window should require a 1-pixel overlap for chunked implementations
        assert neighbours.overlap == (1, 1)

    def test_gridneighb__circular_window(self) -> None:
        """Checks that a three-cell circle includes diagonal cell centers within its radius of 1.5 pixels."""

        # All nine cell centers lie within 1.5 pixels of the middle cell
        neighborhood = GridNeighbours(size=3, shape="circular")

        # Check diagonal cells and the complete window
        assert len(neighborhood.offsets) == 9
        assert (-1, -1) in neighborhood.offsets
        assert (1, 1) in neighborhood.offsets
        assert neighborhood.window_shape == "circular"
        assert neighborhood.overlap == (1, 1)

    def test_gridneighb__asymmetric_offsets(self) -> None:
        """Checks custom offsets for order and overlap per axis."""

        # Asymmetric offsets to check maximum displacement on each axis
        offsets = ((1, 2), (-3, -4), (0, 0))
        neighborhood = GridNeighbours(offsets)
        assert neighborhood.offsets == offsets
        assert neighborhood.overlap == (3, 4)
        assert neighborhood.window_shape is None

    @pytest.mark.parametrize("shape", ["square", "circular"])
    @pytest.mark.parametrize("size,expected_overlap", [(1, 0), (3, 1), (5, 2), (9, 4)])
    def test_gridneighb__overlap_sizes(
        self, size: int, shape: Literal["square", "circular"], expected_overlap: int
    ) -> None:
        """Checks window overlap is robust with window size."""

        neighborhood = GridNeighbours(size=size, shape=shape)

        # Expected value is the number of cells from center to edge
        # Removing corners for a circle leaves the same reach along rows and columns
        assert neighborhood.overlap == (expected_overlap, expected_overlap)

    @pytest.mark.parametrize(
        "offsets,expected_overlap",
        [
            # Offsets
            pytest.param(
                ((0, 0),),
                (0, 0),
                id="center_only",
            ),
            pytest.param(
                ((-3, 0), (1, 0)),
                (3, 0),
                id="rows_only",
            ),
            pytest.param(
                ((0, -2), (0, 4)),
                (0, 4),
                id="columns_only",
            ),
            # Overlap (largest row and column distances from center)
            pytest.param(
                ((1, 4), (3, 2)),
                (3, 4),
                id="positive_offsets",
            ),
            pytest.param(
                ((-1, -4), (-3, -2)),
                (3, 4),
                id="negative_offsets",
            ),
            pytest.param(
                ((-3, 1), (2, -4), (0, 0)),
                (3, 4),
                id="mixed_sign_offsets",
            ),
        ],
    )
    def test_gridneighb__overlap_offsets(
        self, offsets: tuple[tuple[int, int], ...], expected_overlap: tuple[int, int]
    ) -> None:
        """Checks that overlap is robust with custom offset windows"""

        # Define custom window with offsets
        neighborhood = GridNeighbours(offsets)

        # Expected value is the largest distance from the center for row/col
        assert neighborhood.overlap == expected_overlap

    @pytest.mark.parametrize("size", [0, -1, 2, True, 1.5])
    def test_grid_neighbours__error_invalid_size(self, size: int | float) -> None:
        """Checks that raster window sizes must be positive odd integers: not boolean, negative or floating."""

        with pytest.raises(ValueError, match="positive odd integer"):
            GridNeighbours(size=size)  # type: ignore[arg-type]

    def test_grid_neighbours__error_conflicting_inputs(self) -> None:
        """Checks that a window size cannot be combined with explicit offsets or a stored window shape."""

        # Size and shape determine the window offsets
        with pytest.raises(ValueError, match="either a window size or explicit offsets"):
            GridNeighbours(offsets=((0, 0),), size=3)
        with pytest.raises(ValueError, match="either a window size or explicit offsets"):
            GridNeighbours(size=3, window_shape="square")

    @pytest.mark.parametrize("offsets", [(), ((0, 0), (0, 0)), ((0,),), ((0.5, 1),), ((True, 0),)])
    def test_grid_neighbours__error_invalid_offsets(self, offsets: tuple[tuple[int | float, ...], ...]) -> None:
        """Checks that grid neighborhoods reject empty, repeated, incomplete or noninteger offsets."""

        with pytest.raises(ValueError):
            GridNeighbours(offsets)  # type: ignore[arg-type]


######################################
# 2/ NEIGHBOUR DEFAULTS AND VALIDATION
######################################


# 2.1/ Resolve operator defaults
################################


class TestNeighbourDefaults:
    """Test module for point limits chosen by interpolators without an explicit neighborhood."""

    @pytest.mark.parametrize(
        "operator,expected",
        [
            pytest.param(Nearest(), PointNeighbours(k=1), id="nearest"),
            pytest.param(LocalMeanInterpolator(), PointNeighbours(k=8), id="custom_interpolator"),
            pytest.param(
                Kriging(VariogramModel("gaussian", effective_range=4, partial_sill=1), max_overlap=2),
                PointNeighbours(radius=2),
                id="kriging_radius",
            ),
        ],
    )
    def test_resolve_point_neighbours_for_interpolator__defaults(
        self, operator: Interpolator, expected: PointNeighbours
    ) -> None:
        """Checks that point defaults use one nearest point, eight custom inputs or the kriging search radius."""

        # Resolve limits without assigning the source-specific choice to the reusable operator
        neighborhood = _resolve_point_neighbours_for_interpolator(operator)

        # The next operation may use a different source type
        assert neighborhood == expected
        assert operator.default_neighborhood is None


# 2.2/ Build kriging windows from a physical radius
#################################################


class TestKrigingGridNeighbours:
    """Test module for physical search radii on rectangular, rotated and skewed raster grids."""

    @pytest.mark.parametrize(
        "transform",
        [
            pytest.param(Affine(2, 0, 10, 0, -1, 20), id="rectangular"),
            pytest.param(Affine.rotation(30) * Affine.scale(2, -1), id="rotated"),
            pytest.param(Affine(2, 1, 10, 0, -1, 20), id="skewed"),
        ],
    )
    @pytest.mark.parametrize("aligned_targets", [True, False])
    def test_build_kriging_grid_neighbours__physical_radius(self, transform: Affine, aligned_targets: bool) -> None:
        """Checks that kriging windows include every cell within the physical radius on non-square grids."""

        # A radius of 2.1 fits comfortably inside this independent 11x11 candidate grid
        operator = Kriging(VariogramModel("gaussian", effective_range=3, partial_sill=1), max_overlap=2.1)
        candidates = [(row, column) for row in range(-5, 6) for column in range(-5, 6)]
        coordinates = np.array([transform * (column + 0.5, row + 0.5) for row, column in candidates])
        shifts = [(0.0, 0.0)] if aligned_targets else [(-0.49, -0.49), (-0.49, 0.49), (0.49, -0.49), (0.49, 0.49)]

        # Measure distances in map coordinates, without using the inverse-transform search bounds
        expected = set()
        for row_shift, column_shift in shifts:
            target = np.array(transform * (0.5 + column_shift, 0.5 + row_shift))
            inside = np.linalg.norm(coordinates - target, axis=1) <= 2.1
            expected.update(offset for offset, selected in zip(candidates, inside) if selected)

        # Aligned targets permit exact pruning; shifted targets need a window containing every candidate
        neighborhood = _build_kriging_grid_neighbours(operator, transform, aligned_targets=aligned_targets)
        if aligned_targets:
            assert set(neighborhood.offsets) == expected
        else:
            assert expected.issubset(neighborhood.offsets)

    @pytest.mark.parametrize(
        "transform",
        [
            pytest.param(Affine(1, 2, 0, 2, 4, 0), id="singular"),
            pytest.param(Affine(np.nan, 0, 0, 0, 1, 0), id="nonfinite"),
        ],
    )
    def test_build_kriging_grid_neighbours__error_invalid_transform(self, transform: Affine) -> None:
        """Checks that kriging rejects transforms that cannot map a physical radius to finite cell offsets."""

        # Search geometry can be validated without loading an optional kriging backend
        operator = Kriging(VariogramModel("gaussian", effective_range=2, partial_sill=1))

        with pytest.raises(ValueError, match="invertible finite basis"):
            _build_kriging_grid_neighbours(operator, transform)


# 2.3/ Check interpolation windows and chunk overlap
##################################################


class TestResamplingOverlap:
    """Test module for chunk margins; regular interpolation stencil validation is in test_interpolator.py."""

    @pytest.mark.parametrize(
        "method,expected",
        [
            pytest.param("nearest", 1, id="nearest"),
            pytest.param("linear", 2, id="linear"),
            pytest.param("cubic", 4, id="cubic"),
            pytest.param(Mean(GridNeighbours(size=5)), 3, id="reducer_window"),
            pytest.param(LocalMeanInterpolator(GridNeighbours(((0, 0), (-3, 1)))), 4, id="asymmetric_window"),
        ],
    )
    def test_compute_resampling_overlap__methods(
        self, method: ScipyInterpolationMethod | Interpolator | Reducer, expected: int
    ) -> None:
        """Checks that chunk margins cover the interpolation order or window reach plus the boundary cell."""

        # The additional cell covers targets falling between source cell centers
        overlap = _compute_resampling_overlap(method)
        assert overlap == expected

    @pytest.mark.parametrize("operator", [LocalMeanInterpolator(), Mean()])
    def test_compute_resampling_overlap__error_missing_window(self, operator: Interpolator | Reducer) -> None:
        """Checks that custom methods need a grid window before a finite chunk margin can be calculated."""

        with pytest.raises(ValueError, match="requires a built-in method or a GridNeighbours"):
            _compute_resampling_overlap(operator)


###########################
# 3/ RASTER CELL SELECTION
###########################


# 3.1/ Collect cells at fixed offsets
####################################


class TestGridNeighbourSelection:
    """Test module for raster cells, masks, coordinates and source IDs collected into LocalData."""

    def test_prepare_grid_neighbours_data__edge_and_band_ids(self) -> None:
        """Checks that an edge window excludes outside cells and records band IDs, masks and center distances."""

        # Corner window: four cells inside raster, one masked
        values = np.ma.array(np.arange(9.0).reshape(3, 3), mask=False)
        values.mask[1, 0] = True
        transform = rio.transform.from_origin(0, 3, 1, 1)
        operator = LocalMeanInterpolator(neighborhood=GridNeighbours(size=3))
        inputs, handling, _, _ = _prepare_grid_neighbours_data(
            values,
            transform,
            (0.5, 2.5),
            operator,
            area_or_point=None,
            shift_area_or_point=False,
            nodata_propagation="ignore",
            band=2,
        )

        # Check IDs for band 2 (starting at 9) and masked cell flagged invalid
        assert len(inputs) == 1 and handling == "ignore"
        data = inputs[0]
        np.testing.assert_array_equal(data.values, [0, 1, 3, 4])
        np.testing.assert_array_equal(data.source_ids, [9, 10, 12, 13])
        np.testing.assert_array_equal(data.valid, [True, True, False, True])

        # Check cell centers and distances from target
        assert data.coordinates is not None and data.target is not None and data.distances is not None
        np.testing.assert_array_equal(data.coordinates, [[0.5, 2.5], [1.5, 2.5], [0.5, 1.5], [1.5, 1.5]])
        np.testing.assert_array_equal(data.target, [0.5, 2.5])
        np.testing.assert_allclose(data.distances, [0, 1, 1, np.sqrt(2)], rtol=0, atol=1e-15)

    @pytest.mark.parametrize("operator,expected_values", [(LocalMeanInterpolator(), [0, 2]), (Mean(), [])])
    def test_prepare_grid_neighbours_data__outside_target(
        self, operator: Interpolator | Reducer, expected_values: list[int]
    ) -> None:
        """Checks that an outside target can select interpolation neighbors but cannot reduce a containing cell."""

        # The target lies one column left of the raster, while its 3x3 window reaches the first column
        values = np.arange(4.0).reshape(2, 2)
        transform = rio.transform.from_origin(0, 2, 1, 1)
        inputs, _, _, _ = _prepare_grid_neighbours_data(
            values,
            transform,
            (-0.5, 1.5),
            operator,
            area_or_point=None,
            shift_area_or_point=False,
            nodata_propagation="ignore",
            neighborhood=GridNeighbours(size=3),
        )

        # Reducers require a containing source cell even when some window cells intersect the raster
        assert len(inputs) == 1
        np.testing.assert_array_equal(inputs[0].values, expected_values)


# 3.2/ Measure fractional window coverage
########################################


class TestFractionalWindowData:
    """Test module for fractional cell identifiers and validity; public area reductions are in test_resampling.py."""

    def test_prepare_fractional_window_data__weights_and_band_ids(self) -> None:
        """Checks that a shifted window records cell fractions, missing values and source identifiers for its band."""

        # A one-pixel square spans the last quarter of the first cell and three quarters of the second
        values = np.array([[2.0, np.nan], [10.0, 14.0]])
        transform = rio.transform.from_origin(0, 2, 1, 1)
        points = (np.array([1.25]), np.array([1.5]))
        inputs, _, _ = _prepare_fractional_window_data(
            values,
            transform,
            points,
            area_or_point=None,
            fractional_window=1,
            fractional_shape="square",
            band=2,
        )

        # Four cells in band 1 precede the selected cells; nodata remains available to the reducer's policy
        assert len(inputs) == 1
        data = inputs[0]
        np.testing.assert_array_equal(data.source_ids, [4, 5])
        np.testing.assert_array_equal(data.values, [2, np.nan])
        np.testing.assert_array_equal(data.valid, [True, False])
        assert data.support_weights is not None
        np.testing.assert_array_equal(data.support_weights, [0.25, 0.75])


# 3.3/ Prepare regular interpolation values and weights
######################################################


class TestRegularInterpolationData:
    """Test module for selected interpolation coefficients; public uncertainty comparisons are in test_resampling.py."""

    def test_prepare_regular_interpolation_data__missing_cell_and_band_ids(self) -> None:
        """Checks that bilinear preparation omits a missing cell and normalizes the remaining weights."""

        # At row/column 0.25, bilinear weights are 9/16, 3/16, 3/16 and 1/16
        # Removing the upper-right cell leaves 13/16 of the original weight
        values = np.array([[2.0, np.nan], [10.0, 14.0]])
        transform = rio.transform.from_origin(0, 2, 1, 1)
        _, inputs, handling = _prepare_regular_interpolation_data(
            values,
            transform,
            (0.25, 1.75),
            "linear",
            area_or_point=None,
            shift_area_or_point=False,
            nodata_propagation="ignore",
            band=2,
        )

        # Band 2 identifiers and values must follow the same order as the normalized coefficients
        assert len(inputs) == 1 and handling == "ignore"
        data = inputs[0]
        np.testing.assert_array_equal(data.source_ids, [4, 6, 7])
        np.testing.assert_array_equal(data.values, [2, 10, 14])
        np.testing.assert_array_equal(data.valid, [True, True, True])
        assert data.interpolation_weights is not None
        np.testing.assert_allclose(data.interpolation_weights, [9 / 13, 3 / 13, 1 / 13], rtol=0, atol=1e-15)


###########################
# 4/ POINT CLOUD SELECTION
###########################


# 4.1/ Read coordinates, values and source identifiers
#####################################################


class TestPointSourceData:
    """Test module for usable coordinates, missing values, geometry elevations and unique observation identifiers."""

    @pytest.mark.parametrize("data_name", ["value", None])
    @pytest.mark.parametrize("duplicate_ids", [False, True])
    def test_prepare_point_data__invalid_coordinates_and_source_ids(
        self, data_name: str | None, duplicate_ids: bool
    ) -> None:
        """Checks that point preparation excludes unusable coordinates and gives duplicate labels unique row IDs."""

        # Row 1 has no usable position; row 2 has a position but no finite value
        values = [10.0, 99.0, np.nan, 30.0]
        index = [7, 7, 7, 7] if duplicate_ids else [10, 11, 12, 13]
        points = gpd.GeoDataFrame(
            {"value": values},
            index=index,
            geometry=gpd.points_from_xy([0, np.nan, 2, 3], [0, 1, 2, 3], z=values),
        )

        # Built-in gridding separates finite observations from positions used to propagate nodata
        finite_points, finite_values, positioned_points, positioned_valid = _prepare_point_gridding_data(
            points, data_name
        )
        np.testing.assert_array_equal(finite_points, [[0, 0], [3, 3]])
        np.testing.assert_array_equal(finite_values, [10, 30])
        np.testing.assert_array_equal(positioned_points, [[0, 0], [2, 2], [3, 3]])
        np.testing.assert_array_equal(positioned_valid, [True, False, True])

        # Operators receive nodata observations too, with original row positions when labels are duplicated
        coordinates, observed_values, valid, source_ids = _prepare_point_operator_data(points, data_name)
        expected_ids = [0, 2, 3] if duplicate_ids else [10, 12, 13]
        np.testing.assert_array_equal(coordinates, positioned_points)
        np.testing.assert_array_equal(observed_values, [10, np.nan, 30])
        np.testing.assert_array_equal(valid, positioned_valid)
        np.testing.assert_array_equal(source_ids, expected_ids)


# 4.2/ Build spatial searches
#############################


class TestPointSearchGeometry:
    """
    Test module for target ordering and pixel distance units.

    Radius boundaries are also tested through public gridding and filtering methods.
    """

    def test_build_grid_queries__descending_axes(self) -> None:
        """Checks that grid queries follow supplied rows and columns even when both coordinate axes descend."""

        # Unequal axis lengths reveal swapped row/column order
        x_coords = np.array([4.0, 2.0])
        y_coords = np.array([9.0, 6.0, 1.0])
        queries = _build_grid_queries(x_coords, y_coords)

        # Each row visits X coordinates in their supplied order before advancing to the next Y
        expected = [[4, 9], [2, 9], [4, 6], [2, 6], [4, 1], [2, 1]]
        np.testing.assert_array_equal(queries, expected)

    def test_build_scaled_point_tree__rectangular_pixels(self) -> None:
        """Checks that spatial searches measure distances in output pixels relative to the supplied origin."""

        # Four coordinate units per column and two per row, with an origin away from zero
        coordinates = np.array([[8.0, 6.0], [12.0, 10.0]])
        tree = _build_scaled_point_tree(coordinates, x_start=4, y_start=2, res_x=4, res_y=2)

        # From pixel position (0, 2), the first point is one pixel away and the second is sqrt(8) away
        distances, indexes = tree.query([0.0, 2.0], k=2)
        np.testing.assert_array_equal(indexes, [0, 1])
        np.testing.assert_allclose(distances, [1, np.sqrt(8)], rtol=0, atol=1e-15)


# 4.3/ Collect neighbours for each target
########################################


class TestPointNeighbourSelection:
    """Test module for point searches, physical distance limits and observation identities in LocalData."""

    @pytest.mark.parametrize("limit", ["count", "radius", "both"])
    def test_prepare_point_neighbours_data__selected_observations(
        self, limit: Literal["count", "radius", "both"]
    ) -> None:
        """Checks that count and radius searches return the same nearby observations with their original IDs."""

        # Missing observation on radius boundary to check inclusive selection
        points = gpd.GeoDataFrame(
            {"value": [2.0, np.nan, 100.0]},
            index=[20, 10, 30],
            geometry=gpd.points_from_xy([0, 1, 3], [0, 0, 0]),
            crs=32631,
        )
        neighborhood = PointNeighbours(k=2 if limit != "radius" else None, radius=1 if limit != "count" else None)
        operator = LocalMeanInterpolator(neighborhood=neighborhood)
        inputs, indexes = _prepare_point_neighbours_data(
            points,
            (np.array([0.0]), np.array([0.0])),
            "value",
            operator,
            res_x=2,
            res_y=3,
            radius=5,
            min_points=1,
        )

        # Check selection by coordinate distance with non-square output pixels
        assert indexes == [(0, 0)]
        data = inputs[0]
        np.testing.assert_array_equal(data.source_ids, [20, 10])
        np.testing.assert_array_equal(data.values, [2, np.nan])
        np.testing.assert_array_equal(data.valid, [True, False])
        assert data.distances is not None and data.coordinates is not None
        np.testing.assert_array_equal(data.distances, [0, 1])
        np.testing.assert_array_equal(data.coordinates, [[0, 0], [1, 0]])

    @pytest.mark.parametrize(
        "x,values",
        [
            pytest.param([], [], id="empty_source"),
            pytest.param([np.nan], [2.0], id="no_usable_coordinates"),
            pytest.param([0.0], [np.nan], id="no_finite_values"),
            pytest.param([2.0], [2.0], id="outside_support_radius"),
        ],
    )
    def test_prepare_point_neighbours_data__no_supported_targets(self, x: list[float], values: list[float]) -> None:
        """Checks that empty, unusable or distant sources produce no nearest-neighbor inputs or output indexes."""

        # A target at the origin needs a finite observation within one output pixel
        points = gpd.GeoDataFrame({"value": values}, geometry=gpd.points_from_xy(x, np.zeros(len(x))))
        grid_coords = (np.array([0.0]), np.array([0.0]))
        inputs, indexes = _prepare_point_neighbours_data(
            points,
            grid_coords,
            "value",
            Nearest(),
            res_x=1,
            res_y=1,
            radius=1,
            min_points=1,
        )

        # No selected group should be passed to the operator or assigned an output cell
        assert inputs == []
        assert indexes == []


# 4.4/ Grid within a radius and mask unsupported cells
#####################################################


class TestRadiusGridding:
    """Test module for bounded row searches; method comparisons and lazy execution are in test_gridding.py."""

    @pytest.mark.parametrize("method", ["mean", "idw", "nearest"])
    def test_grid__last_row_block(self, method: Literal["mean", "idw", "nearest"]) -> None:
        """Checks that gridding and support masks place observations correctly across a 128-row search boundary."""

        # A 129-row grid leaves one row in the final search block, with observations on both sides of its boundary
        points = gu.PointCloud.from_xyz([0, 0, 0], [0, 127, 128], [2.0, 4.0, 8.0], crs=32631)
        reference = gu.Raster.from_array(np.zeros((129, 2)), rio.transform.from_origin(0, 128, 1, 1), crs=32631)
        result = points.grid(
            ref=reference,
            resampling=method,
            dist_nodata_pixel=0.1,
            nodata_handling="ignore",
            engine="scipy",
        )

        # The north-up raster reverses the ascending Y search order; only coincident targets have support
        expected = np.full((129, 2), np.nan)
        expected[[128, 1, 0], 0] = [2, 4, 8]
        np.testing.assert_array_equal(result.to_nanarray(), expected)
