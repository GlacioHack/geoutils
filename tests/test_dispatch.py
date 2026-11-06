"""
Test the dispatch functions used for checking and normalizing inputs.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import pytest
import rasterio as rio
from shapely.geometry import Point, Polygon

import geoutils as gu
from geoutils._dispatch import (  # Level-1 checks (match reference object)
    _check_bounds,
    _check_coords,  # Level-0 checks (manual input)
    _check_crs,
    _check_match_bbox,
    _check_match_grid,
    _check_match_points,
    _check_nodata,
    _check_nodata_propagation,
    _check_resampling,
    _check_resolution,
    _check_same_crs,
    _check_same_grid,
    _check_same_points,
    _check_same_shape,
    _check_shape,
    _check_valid_data,
    _clip_geodataframe,
    _get_reproject_crs,
    _grid_from_bounds_res,
    _grid_from_bounds_shape,
    _grid_from_coords,
    _grid_from_src,  # Helpers for grid conversion
    get_geo_interface,
)
from geoutils.exceptions import (
    CRSMismatchError,
    GeoUtilsInputError,
    GridMismatchError,
    IgnoredGridWarning,
    InvalidBoundsError,
    InvalidCRSError,
    InvalidGridError,
    InvalidNodataError,
    InvalidPointsError,
    InvalidResamplingError,
    InvalidResolutionError,
    InvalidShapeError,
    NoValidDataError,
    PointsMismatchError,
    ShapeMismatchError,
)


class TestGeoInterface:
    """Test module for spatial metadata shared by an object or accessor and its geospatial operations."""

    @pytest.mark.parametrize("representation", ["raster", "accessor", "dataarray"])
    def test_get_geo_interface__raster_metadata(self, representation: str) -> None:
        """Checks that raster band count and spatial shape stay distinct from Xarray methods and dimensions."""

        # Include two bands so Xarray shape differs from the two-dimensional raster grid
        raster = gu.Raster.from_array(np.ones((2, 3, 4)), rio.transform.from_origin(0, 3, 1, 1), crs=32633)
        array = raster.to_xarray()
        source = {"raster": raster, "accessor": array.rst, "dataarray": array}[representation]

        # Resolve one interface and read both metadata fields from it
        interface = get_geo_interface(source, "ij2xy", accessors=("rst",))
        assert interface.count == 2
        assert interface.shape == (3, 4)
        assert interface.transform == raster.transform

        # Plain arrays have no spatial interface and cannot supply a reference grid
        assert get_geo_interface(np.ones((3, 4)), "ij2xy", accessors=("rst",)) is None

    @pytest.mark.parametrize("representation", ["vector", "accessor", "geodataframe"])
    def test_get_geo_interface__vector_metadata(self, representation: str) -> None:
        """Checks that vector objects and GeoDataFrames provide the vector interface and its bounds and CRS."""

        frame = gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (2, 0), (2, 3), (0, 3)])], crs=32633)
        vector = gu.Vector(frame)
        source = {"vector": vector, "accessor": frame.vct, "geodataframe": frame}[representation]

        interface = get_geo_interface(source, "rasterize")

        assert interface is (frame.vct if representation == "geodataframe" else source)
        assert interface.bbox == (0, 0, 2, 3)
        assert interface.crs == pyproj.CRS(32633)

    @pytest.mark.parametrize(
        "representation", ["pointcloud", "accessor", "geodataframe", "dataarray_accessor", "dataarray"]
    )
    def test_get_geo_interface__point_metadata(self, representation: str) -> None:
        """Checks that point objects, GeoDataFrames and DataArrays provide point count, bounds and CRS."""

        points = gu.PointCloud.from_xyz([0, 1, 2], [2, 0, 1], [4, 5, 6], crs=32633)
        array = points.to_xarray()
        sources = {
            "pointcloud": points,
            "accessor": points.gdf.pc,
            "geodataframe": points.gdf,
            "dataarray_accessor": array.pc,
            "dataarray": array,
        }

        # The point method distinguishes the pc accessor from the shared vector operations
        interface = get_geo_interface(sources[representation], "georeferenced_coords_equal")

        if representation == "dataarray":
            assert interface is array.pc
        elif representation == "geodataframe":
            assert interface is points.gdf.pc
        else:
            assert interface is sources[representation]
        assert interface.point_count == 3
        assert interface.bbox == (0, 0, 2, 2)
        assert interface.crs == pyproj.CRS(32633)

class TestClipGeometry:
    """Test module for preserving clipping features and matching their coordinate reference system."""

    def test_clip_geodataframe__keeps_features_in_target_crs(self) -> None:
        """Checks that clipping normalization reprojects features without dissolving them."""

        # Create two disconnected polygons whose separate rows can later be assigned to different raster chunks
        source = gpd.GeoDataFrame(
            geometry=[
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
            ],
            crs=4326,
        )

        # Normalize the mask to a different target CRS through the same helper used by raster clip()
        normalized = _clip_geodataframe(source, target_crs=rio.crs.CRS.from_epsg(3857))

        # Both rows remain available for spatial partitioning and their coordinates match an independent reprojection
        assert len(normalized) == 2
        assert normalized.crs == "EPSG:3857"
        gpd.testing.assert_geodataframe_equal(normalized, source.to_crs(3857))


class TestDispatchLevelZero:
    """Level zero user-input checks: CRS, bounds, shape, resolution, regular coordinates, point coordinates."""

    def test_check_crs__valid(self) -> None:
        """Check that valid CRS input pass."""

        # Valid CRS string
        crs = _check_crs("EPSG:4326")
        assert isinstance(crs, pyproj.CRS)

        # Valid EPSG int
        crs2 = _check_crs(4326)
        assert isinstance(crs2, pyproj.CRS)

        # Valid CRS object
        crs3 = _check_crs(pyproj.CRS.from_epsg(4326))
        assert isinstance(crs3, pyproj.CRS)

        # They should all be the same
        assert crs == crs2
        assert crs2 == crs3

    def test_check_crs__exceptions(self) -> None:
        """Check that invalid CRS input raise proper errors."""

        # Raised by Pyproj and relayed correctly
        with pytest.raises(InvalidCRSError, match="Projection not recognized.*"):
            _check_crs("lol")

    def test_check_bounds__valid(self) -> None:
        """Check that valid bounds input pass."""

        # Valid sequence (typing is tuple of 4 for readability, but function accepts any sequence)
        bounds = [0, 0, 10, 10]
        xmin, ymin, xmax, ymax = _check_bounds(bounds)  # type: ignore
        assert (xmin, ymin, xmax, ymax) == (0.0, 0.0, 10.0, 10.0)

        # Valid bounding box
        bbox = rio.coords.BoundingBox(left=1, bottom=2, right=3, top=4)
        xmin, ymin, xmax, ymax = _check_bounds(bbox)
        assert (xmin, ymin, xmax, ymax) == (1, 2, 3, 4)

        # Valid dataframe (bounding box of geodataframe)
        bbox = pd.DataFrame(data={"minx": 0, "miny": 1, "maxx": 2, "maxy": 3}, index=[7])
        xmin, ymin, xmax, ymax = _check_bounds(bbox)
        assert (xmin, ymin, xmax, ymax) == (0, 1, 2, 3)

        # Arrays and point extents use the same normalization
        assert _check_bounds(np.array([0, 1, 2, 3])) == (0, 1, 2, 3)
        assert _check_bounds((1, 2, 1, 2)) == (1, 2, 1, 2)

        # Valid dataframe (bounding box of geodataframe)
        bbox = {"left": 0, "bottom": 1, "right": 2, "top": 3}
        xmin, ymin, xmax, ymax = _check_bounds(bbox)
        assert (xmin, ymin, xmax, ymax) == (0, 1, 2, 3)

    @pytest.mark.parametrize(
        "bbox, match_text",
        [
            ([0, 1, 2], "length of 3"),  # Invalid length
            ([5, 0, 2, 4], "xmin must be <= xmax"),  # Invalid ordering
            ([0, 5, 4, 2], "ymin must be <= ymax"),  # Invalid ordering
            (rio.coords.BoundingBox(5, 0, 2, 4), "xmin must be <= xmax"),
            ({"left": 5, "bottom": 0, "right": 2, "top": 4}, "xmin must be <= xmax"),
            (pd.DataFrame({"minx": [5], "miny": [0], "maxx": [2], "maxy": [4]}), "xmin must be <= xmax"),
            ((0, 0, np.nan, 1), "must be finite"),
            (rio.coords.BoundingBox(0, 0, 1, np.inf), "must be finite"),
            (np.zeros((2, 2)), "one-dimensional"),
            (pd.DataFrame(data={"notgood": [1]}), "must contain columns"),  # Invalid columns in dataframe
            # Too lengthy inside dataframe
            (
                pd.DataFrame(data={"minx": [0, 1], "miny": [1, 2], "maxx": [2, 3], "maxy": [3, 4]}),
                "must contain columns",
            ),
            ({"otherkey": 5}, "should have keys"),  # Wrong key
            ({"left": "a", "bottom": 0, "right": 1, "top": 3}, "value for 'left' must be numeric"),  # Key non-numeric
            (["a", "b", "c", "d"], "must be numeric"),  # Non-numeric
            (42, "Cannot interpret bounding box input"),  # Invalid type
        ],
    )
    def test_check_bounds__exceptions(self, bbox: Any, match_text: str) -> None:
        """Checks an error is raised for invalid bounds in each supported representation."""

        with pytest.raises(GeoUtilsInputError, match=match_text) as error:
            _check_bounds(bbox)  # type: ignore
        assert isinstance(error.value, InvalidBoundsError)
        assert isinstance(error.value, ValueError)

    def test_check_resolution__valid(self) -> None:
        """Check that valid resolution input pass."""

        # Valid scalar resolution, normalized to tuple
        assert _check_resolution(5) == (5, 5)

        # Valid sequence resolution, normalized to tuple
        assert _check_resolution([1.0, 2.0]) == (1.0, 2.0)  # type: ignore
        assert _check_resolution(np.array([1.0, 2.0])) == (1.0, 2.0)

    @pytest.mark.parametrize(
        "res, match_text",
        [
            ([1, 2, 3], "sequence of length 3"),  # Invalid length
            (["a", "b"], "must be numeric"),  # Non-numeric in sequence
            ([-1, 1], "strictly positive"),  # Non-positive in sequence
            ([2, np.nan], "finite number"),  # Non-finite in sequence
            (["a", "b"], "must be numeric"),  # Non-numeric
            (0, "strictly positive"),  # Non-positive for scalar
            (np.inf, "finite number"),  # Non-finite for scalar
            ("wrong", "Resolution must be a number or a sequence"),  # Non-finite for scalar
            ("12", "Resolution must be a number or a sequence"),
            (True, "not a boolean"),
            (np.bool_(True), "not a boolean"),
            ((True, 2), "not booleans"),
            (np.array(2), "one-dimensional"),
        ],
    )
    def test_check_resolution__exceptions(self, res: Any, match_text: str) -> None:
        """Check that invalid resolution input raise proper errors."""

        with pytest.raises(InvalidResolutionError, match=match_text):
            _check_resolution(res)  # type: ignore

    def test_check_shape__valid(self) -> None:
        """Check that valid shape input pass."""

        # Valid shape as tuple
        assert _check_shape((10, 20)) == (10, 20)
        # Valid shape as other sequence
        assert _check_shape([10, 20]) == (10, 20)  # type: ignore
        assert _check_shape((np.int64(10), np.int64(20))) == (10, 20)
        assert _check_shape((0, 20)) == (0, 20)

    @pytest.mark.parametrize(
        "shape, match_text",
        [
            (10, "sequence of two integers"),  # Not a sequence
            ((10,), "length 1"),  # Wrong length
            ((-1, 5), "non-negative"),  # Negative
            (("a", 5), "must be integers"),  # Non-integer
            ((2.8, 3), "must be integers"),
            ((np.inf, 3), "must be integers"),
            ((np.nan, 3), "must be integers"),
            ((True, 3), "must be integers"),
            ((np.bool_(True), 3), "must be integers"),
            (("2", 3), "must be integers"),
            (np.array(3), "one-dimensional"),
        ],
    )
    def test_check_shape__exceptions(self, shape: Any, match_text: str) -> None:
        """Check that invalid shape input raise proper errors."""

        with pytest.raises(InvalidShapeError, match=match_text):
            _check_shape(shape)

    @pytest.mark.parametrize(
        "coords, expected_dxdy",
        [
            # Regular increasing
            (([0, 1, 2, 3], [10, 11, 12, 13]), (1.0, 1.0)),
            # Regular non-unit spacing
            (([0, 2, 4, 6], [0, 5, 10, 15]), (2.0, 5.0)),
            # Regular decreasing
            (([3, 2, 1, 0], [10, 8, 6, 4]), (-1.0, -2.0)),
            # NumPy arrays
            ((np.array([0.0, 0.5, 1.0]), np.array([10.0, 20.0, 30.0])), (0.5, 10.0)),
        ],
    )
    def test_check_coords__valid(self, coords: Any, expected_dxdy: Any) -> None:
        """Check that valid regular coordinates pass and return resolution."""
        (x, y), (dx, dy) = _check_coords(coords)

        assert np.array_equal(x, np.asarray(coords[0], dtype=float))
        assert np.array_equal(y, np.asarray(coords[1], dtype=float))
        assert dx == pytest.approx(expected_dxdy[0])
        assert dy == pytest.approx(expected_dxdy[1])

    @pytest.mark.parametrize(
        "coords, match_text",
        [
            # Not a sequence
            (42, "must be a sequence of two array-like objects"),
            ("abc", "must be a sequence of two array-like objects"),
            # Wrong length
            (([0, 1, 2],), "length of 1"),
            (([0, 1], [2, 3], [4, 5]), "length of 3"),
            # Non-numeric
            ((["a", "b"], [0, 1]), "X coordinates must be numeric"),
            (([0, 1], ["x", "y"]), "Y coordinates must be numeric"),
            # Not 1D
            (([[0, 1], [2, 3]], [0, 1]), "must be 1D"),
            (([0, 1], [[0, 1], [2, 3]]), "must be 1D"),
            # Too small (size < 2)
            (([0], [0]), "must contain at least 2 points"),
            # Irregular spacing (x)
            (([0, 1, 3], [0, 1, 2]), "must be regular"),
            # Irregular spacing (y)
            (([0, 1, 2], [0, 1, 3]), "must be regular"),
            (([0, 0, 0], [0, 1, 2]), "strictly increasing or decreasing"),
            (([0, np.nan], [0, 1]), "must be finite"),
            (([0, 1], [0, np.inf]), "must be finite"),
            (([-1e308, 1e308], [0, 1]), "spacing must be finite"),
            (([0, 1e-10, 0], [0, 1, 2]), "strictly increasing or decreasing"),
        ],
    )
    def test_check_coords__exceptions(self, coords: Any, match_text: str) -> None:
        """Check that invalid coordinate inputs raise proper errors."""
        with pytest.raises(InvalidGridError, match=match_text):
            _check_coords(coords)

    @pytest.mark.parametrize(
        "nodata,dtype,allow_cast,compatible",
        [
            (None, "uint8", False, True),
            (255, "uint8", False, True),
            (np.nan, "float32", False, True),
            (np.float32(-9999), None, False, True),
            (-1, "uint8", True, False),
            (np.nan, "uint8", True, False),
        ],
    )
    def test_check_nodata__dtype(self, nodata: Any, dtype: str | None, allow_cast: bool, compatible: bool) -> None:
        """Checks that nodata compatibility accounts for NaN and allowed dtype changes."""

        assert _check_nodata(nodata, dtype, allow_cast=allow_cast) is compatible

    @pytest.mark.parametrize("method,allowed", [("nearest", ("nearest", "linear")), ("scipy", ("scipy", "numba"))])
    def test_check_resampling__valid(self, method: str, allowed: tuple[str, ...]) -> None:
        """Checks that available resampling methods and backends pass."""

        assert _check_resampling(method, allowed) == method
        assert _check_resampling("nearest") == "nearest"

    @pytest.mark.parametrize("rule", ["nearest", "IGNORE", "Propagate"])
    def test_check_nodata_propagation__valid(self, rule: str) -> None:
        """Checks that nodata propagation rules are normalized to lowercase."""

        assert _check_nodata_propagation(rule) == rule.lower()

    @pytest.mark.parametrize("count,minimum", [(1, 1), (2, 2), (3, 2)])
    def test_check_valid_data__minimum(self, count: int, minimum: int) -> None:
        """Checks that enough valid observations pass validation."""

        _check_valid_data(count, minimum)

    @pytest.mark.parametrize("nodata,dtype", [("missing", None), ([0], None), (-1, "uint8"), (np.nan, "uint8")])
    def test_check_nodata__error_invalid_value(self, nodata: Any, dtype: str | None) -> None:
        """Checks an error is raised for invalid nodata or an incompatible dtype."""

        with pytest.raises(InvalidNodataError, match="nodata|Nodata"):
            _check_nodata(nodata, dtype)

    @pytest.mark.parametrize("method", ["unknown", None, 1])
    def test_check_resampling__error_unknown_method(self, method: Any) -> None:
        """Checks an error is raised for an unknown resampling method."""

        with pytest.raises(InvalidResamplingError, match="Unknown resampling method"):
            _check_resampling(method)

    @pytest.mark.parametrize("rule", ["gdal", "unknown", None, 1])
    def test_check_nodata_propagation__error_invalid_rule(self, rule: Any) -> None:
        """Checks an error is raised for an invalid nodata propagation rule."""

        with pytest.raises(InvalidNodataError, match="nodata_propagation must be one of"):
            _check_nodata_propagation(rule)

    @pytest.mark.parametrize("count,minimum", [(0, 1), (1, 2)])
    def test_check_valid_data__error_insufficient_observations(self, count: int, minimum: int) -> None:
        """Checks an error is raised with the operation's message for too few valid observations."""

        with pytest.raises(NoValidDataError, match="Too few observations"):
            _check_valid_data(count, minimum, message="Too few observations")


class TestMatchingInputs:
    """Test module for checks on matching inputs."""

    @pytest.mark.parametrize("crs", [None, pyproj.CRS(32631), rio.crs.CRS.from_epsg(32631)])
    def test_check_same_crs__equal(self, crs: Any) -> None:
        """Checks that equal CRSs and matching absent CRS metadata pass validation."""

        _check_same_crs(crs, crs)

    @pytest.mark.parametrize("shape", [(), (2,), (2, 3), (2, 3, 4)])
    def test_check_same_shape__equal(self, shape: tuple[int, ...]) -> None:
        """Checks that matching array shapes pass regardless of their number of dimensions."""

        _check_same_shape(shape, shape)

    @pytest.mark.parametrize("representation", ["coordinates", "pointcloud", "dataarray"])
    def test_check_same_points__ordered_coordinates(self, representation: str) -> None:
        """Checks that matching point coordinates pass and reordered points raise a mismatch error."""

        coordinates = np.array([[0, 1], [2, 3], [4, 5]], dtype=float)
        points: Any
        other: Any
        if representation == "coordinates":
            points, other = coordinates, coordinates[::-1]
        else:
            points = gu.PointCloud.from_xyz(coordinates[:, 0], coordinates[:, 1], np.ones(3), crs=32631)
            other = gu.PointCloud.from_xyz(coordinates[::-1, 0], coordinates[::-1, 1], np.ones(3), crs=32631)
            if representation == "dataarray":
                points, other = points.to_xarray(), other.to_xarray()

        _check_same_points(points, points)
        with pytest.raises(PointsMismatchError, match="Different points"):
            _check_same_points(points, other, "Different points")

    def test_check_same_crs__axis_order(self) -> None:
        """Checks that CRS comparison ignores axis order only when requested."""

        # EPSG:4326 and CRS84 use opposite axis orders
        crs, other = pyproj.CRS(4326), pyproj.CRS("OGC:CRS84")
        with pytest.raises(CRSMismatchError):
            _check_same_crs(crs, other)
        _check_same_crs(crs, other, ignore_axis_order=True)

    def test_check_same_grid__unloaded(self, tmp_path: Path) -> None:
        """Checks that matching grids are compared without loading raster pixels."""

        # Write and open two unloaded rasters
        path = tmp_path / "grid.tif"
        gu.Raster.from_array(np.ones((2, 3)), rio.transform.from_origin(0, 2, 1, 1), 32631).to_file(path)
        first, second = gu.Raster(path), gu.Raster(path)

        _check_same_grid(first, second)

        assert not first.is_loaded
        assert not second.is_loaded

    @pytest.mark.parametrize("representation", ["raster", "dataarray"])
    def test_check_same_grid__error_mismatch(self, representation: str) -> None:
        """Checks an error is raised for different grids through either raster interface."""

        first = gu.Raster.from_array(np.ones((2, 3)), rio.transform.from_origin(0, 2, 1, 1), 32631)
        second = first.translate(xoff=1, yoff=0)
        other = second if representation == "raster" else second.to_xarray()

        with pytest.raises(GridMismatchError, match="Different grids"):
            _check_same_grid(first, other, "Different grids")

    def test_check_same_crs__error_invalid_definition(self) -> None:
        """Checks an error is raised for an invalid CRS before comparing inputs."""

        with pytest.raises(InvalidCRSError):
            _check_same_crs("invalid-crs", pyproj.CRS(32631))

    @pytest.mark.parametrize("shapes", [((2,), (3,)), ((2, 3), (3, 2)), ((2, 3), (1, 2, 3))])
    def test_check_same_shape__error_mismatch(self, shapes: tuple[tuple[int, ...], tuple[int, ...]]) -> None:
        """Checks an error is raised for different array sizes or dimensions."""

        with pytest.raises(ShapeMismatchError, match="Different shapes"):
            _check_same_shape(*shapes, message="Different shapes")

    def test_check_same_points__error_crs_mismatch(self) -> None:
        """Checks an error is raised for different point CRSs even when X/Y values match."""

        first = gu.PointCloud.from_xyz(np.arange(3), np.arange(3), np.ones(3), crs=32631)
        second = gu.PointCloud.from_xyz(np.arange(3), np.arange(3), np.ones(3), crs=32632)

        with pytest.raises(CRSMismatchError):
            _check_same_points(first, second)

    @pytest.mark.parametrize("object_type", ["raster", "pointcloud"])
    def test_matching_inputs__vertical_crs_warning(self, object_type: str) -> None:
        """Checks that equal horizontal coordinates with different vertical CRSs warn once and pass."""

        first = gu.Raster.from_array(np.ones((2, 3)), rio.transform.from_origin(0, 2, 1, 1), 4326)
        second = first.copy()
        second.set_crs(4979)

        # EPSG:4979 adds height to EPSG:4326
        with pytest.warns(UserWarning, match="same 2D CRS") as caught:
            if object_type == "raster":
                _check_same_grid(first, second)
            else:
                _check_same_points(first.to_pointcloud(), second.to_pointcloud())
        assert len(caught) == 1


class TestDispatchGridHelpers:
    """Helpers to build grid from inputs."""

    # Raster and vector classes
    rast_bounds = (0, 0, 10, 10)
    rast_res = (2, 2)
    rast_half_res = (rast_res[0] / 2, rast_res[1] / 2)
    rast_shape = (5, 5)
    rast_twice_shape = (rast_shape[0] * 2, rast_shape[1] * 2)
    rast = gu.Raster.from_array(np.zeros((5, 5)), transform=rio.transform.from_bounds(0, 0, 10, 10, 5, 5), crs=4326)
    vect = gu.Vector(gpd.GeoDataFrame({"geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]}, crs="EPSG:4326"))

    def test_grid_from_coords__valid(self) -> None:
        """Check that valid coordinates input pass."""

        # Valid grid coordinates
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        shape, tfm = _grid_from_coords((x, y))
        assert isinstance(shape, tuple)
        assert len(shape) == 2
        assert isinstance(shape[0], int) and isinstance(shape[1], int)
        assert isinstance(tfm, rio.Affine)
        assert np.isclose(tfm.a, x[1] - x[0])
        assert np.isclose(tfm.e, y[0] - y[1])  # Negative northing

    def test_grid_from_coords__rectangular_centers(self) -> None:
        """Checks that rectangular coordinate axes correspond to cell centers with separate X/Y resolutions."""

        # Create axes with different lengths and spacings
        x = np.array([105, 115, 125, 135])
        y = np.array([210, 230, 250])

        # Build the transform from center coordinates
        shape, transform = _grid_from_coords((x, y))
        raster = gu.Raster.from_array(np.zeros(shape), transform, crs=32632)
        actual_x, actual_y = raster.coords(grid=False)

        # Check array dimensions, bounds and the original center
        assert shape == (3, 4)
        assert raster.bounds == (100, 200, 140, 260)
        np.testing.assert_array_equal(actual_x, x)
        np.testing.assert_array_equal(actual_y, y)

    def test_grid_from_coords__exceptions(self) -> None:
        """Check that invalid coordinates input raise proper errors."""

        # Invalid irregular coordinates
        x = np.array([0, 1, 2, 4])  # Irregular
        y = np.linspace(0, 1, 4)
        with pytest.raises(InvalidGridError, match="equally space"):
            _grid_from_coords((x, y))

        # Too short coordinates (need 2 points at least)
        x = np.array([0])
        y = np.array([0])
        with pytest.raises(InvalidGridError, match="at least 2 points"):
            _grid_from_coords((x, y))

    def test_grid_from_bounds_res_and_shape(self) -> None:
        """Test that the conversion works as intended (no errors to check here, depends on above)."""

        # Valid bounds and resolution conversion
        bounds = (0, 0, 10, 20)
        res = (2, 5)
        shape, tfm = _grid_from_bounds_res(bounds, res)
        assert shape == (4, 5)
        assert isinstance(tfm, rio.Affine)

        # Valid bounds and shape conversion
        shape2 = (5, 4)
        shape, tfm2 = _grid_from_bounds_shape(bounds, shape2)
        assert isinstance(tfm2, rio.Affine)

    @pytest.mark.parametrize("shape", [(0, 3), (3, 0)])
    def test_grid_from_bounds_shape__error_empty_grid(self, shape: tuple[int, int]) -> None:
        """Checks an error is raised before constructing a target grid with an empty dimension."""

        with pytest.raises(InvalidShapeError, match="positive integers"):
            _grid_from_bounds_shape((0, 0, 3, 3), shape)

    def test_grid_from_bounds_res__error_empty_grid(self) -> None:
        """Checks an error is raised when the resolution would round a target grid down to zero cells."""

        with pytest.raises(InvalidShapeError, match="positive integers"):
            _grid_from_bounds_res((0, 0, 1, 1), 10)

    @pytest.mark.parametrize(
        "dst_crs,src_kind,shape,res,bounds,exp_shape,exp_same_tfm",
        [
            # Same CRS raster = exact passthrough
            (4326, "rast", None, None, None, rast_shape, True),
            # Same CRS raster + same shape = still passthrough
            (4326, "rast", rast_shape, None, None, (5, 5), True),
            # Same CRS raster + same resolution = still passthrough
            (4326, "rast", None, rast_res, None, (5, 5), True),
            # Same CRS raster + same bounds = still passthrough
            (4326, "rast", None, None, rast_bounds, (5, 5), True),
            # Same CRS raster + same shape/bounds = still passthrough
            (4326, "rast", rast_shape, None, rast_bounds, (5, 5), True),
            # Same CRS raster + same res/bounds = still passthrough
            (4326, "rast", None, rast_res, rast_bounds, (5, 5), True),
            # Same CRS raster + half-resolution = twice shape (with bounds or not)
            (4326, "rast", None, rast_half_res, None, rast_twice_shape, False),
            (4326, "rast", None, rast_half_res, rast_bounds, rast_twice_shape, False),
            # Same CRS raster + twice shape = twice shape (with bounds or not)
            (4326, "rast", rast_twice_shape, None, None, rast_twice_shape, False),
            (4326, "rast", rast_twice_shape, None, rast_bounds, rast_twice_shape, False),
            # CRS change raster = default reprojection grid
            (3857, "rast", None, None, None, (5, 5), False),
            # CRS change raster + explicit shape = force shape
            (3857, "rast", (6, 6), None, None, (6, 6), False),
            # CRS change raster + explicit resolution = force resolution
            (3857, "rast", None, (1000, 1000), None, None, False),  # Can't know output shape when changing resolution
            # CRS change raster + bounds + shape = bounds must match exactly
            (3857, "rast", (4, 4), None, (0, 0, 5, 5), (4, 4), False),
            # CRS change raster + bounds + resolution = infer shape from bounds
            (3857, "rast", None, (1000, 1000), (0, 0, 5000, 5000), None, False),  # Same here
            # Vector source + explicit shape = grid built from bounds
            (3857, "vect", (4, 4), None, None, (4, 4), False),
            # Vector source + explicit resolution = grid built from bounds
            (3857, "vect", None, (1000, 1000), None, None, False),  # Same here
        ],
    )
    def test_grid_from_src__valid(
        self,
        dst_crs: int,
        src_kind: str,
        shape: tuple[int, int] | None,
        res: tuple[float, float] | None,
        bounds: tuple[float, float, float, float] | None,
        exp_shape: tuple[int, int] | None,
        exp_same_tfm: bool,
    ) -> None:
        """Check that valid inputs for grid from source pass and give the right outputs."""

        # Get source (raster or vector)
        src = self.rast if src_kind == "rast" else self.vect
        crs = pyproj.CRS.from_user_input(dst_crs)

        # Compute output shape and transform
        out_shape, out_transform = _grid_from_src(dst_crs=crs, src=src, shape=shape, res=res, bounds=bounds)

        # If expected shape is passed, check it matches expected
        if exp_shape is not None:
            assert out_shape == exp_shape
        # Check transform is affine
        assert isinstance(out_transform, rio.Affine)

        # If bounds were passed, check it matches exactly
        if bounds is not None:
            xmin, ymin, xmax, ymax = rio.transform.array_bounds(*out_shape, out_transform)
            assert (xmin, ymin, xmax, ymax) == pytest.approx(bounds)

        # If res was passed, check it matches exactly
        if res is not None:
            rx, ry = res
            a, b, _, d, e, _ = out_transform[:6]

            assert b == 0
            assert d == 0
            assert a == pytest.approx(rx)
            assert -e == pytest.approx(ry)

        # Check if we expect the exact same transform, or not
        if exp_same_tfm:
            assert out_transform == src.transform
        else:
            assert out_transform != src.transform or crs != src.crs


class TestDispatchLevelOne:
    """Level one user-input checks: match bbox, match points and match grid."""

    rast = gu.Raster.from_array(np.zeros((5, 5)), transform=rio.transform.from_bounds(0, 0, 10, 10, 5, 5), crs=4326)
    rast2 = gu.Raster.from_array(np.zeros((5, 5)), transform=rio.transform.from_bounds(0, 0, 5, 5, 5, 5), crs=4326)
    vect = gu.Vector(gpd.GeoDataFrame({"geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]}, crs="EPSG:4326"))
    vect2 = gu.Vector(gpd.GeoDataFrame({"geometry": [Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])]}, crs="EPSG:4326"))
    # Xarray DataArray and Vector geodataframe
    rast_xr = rast.to_xarray()
    vect_gdf = vect.gdf

    @pytest.mark.parametrize(
        "bbox_input, expected",
        [
            # Sequence inputs
            ([0, 0, 10, 20], (0, 0, 10, 20)),
            ((1.0, 2.0, 5.0, 6.0), (1.0, 2.0, 5.0, 6.0)),
            # Rasterio BoundingBox
            (rio.coords.BoundingBox(left=0, bottom=0, right=10, top=20), (0, 0, 10, 20)),
            # Raster-like objects
            (rast, (0, 0, 10, 10)),
            (rast_xr, (0, 0, 10, 10)),
            # Vector-like objects
            (vect, (0, 0, 1, 1)),
            (vect_gdf, (0, 0, 1, 1)),
            # Third-party object using the common bbox name without a bounds alias
            (SimpleNamespace(bbox=(2, 3, 4, 5), crs=pyproj.CRS.from_epsg(4326)), (2, 3, 4, 5)),
            # Third-party object retaining the previous bounds name
            (SimpleNamespace(bounds=(3, 4, 5, 6), crs=pyproj.CRS.from_epsg(4326)), (3, 4, 5, 6)),
        ],
    )
    def test_check_match_bbox__valid(self, bbox_input: Any, expected: Any) -> None:
        """Check that valid match-bbox input pass."""

        xmin, ymin, xmax, ymax = _check_match_bbox(src=self.rast, bbox=bbox_input)
        assert (xmin, ymin, xmax, ymax) == expected

    @pytest.mark.parametrize(
        "bbox_input, match_text",
        [
            # Wrong length sequences
            ([0, 0, 10], "length of 3"),
            ((1, 2, 3), "length of 3"),
            # Non-numeric sequence
            (["a", 0, 1, 2], "must be numeric"),
            ([0, None, 1, 2], "must be numeric"),
            # xmin >= xmax
            ([10, 0, 5, 20], "xmin.*<.*xmax"),
            # ymin >= ymax
            ([0, 20, 10, 5], "ymin.*<.*ymax"),
            # Totally invalid type
            (42, "Cannot interpret bounding box input"),
            ("abcd", "Cannot interpret bounding box input"),
            (None, "Cannot interpret bounding box input"),
        ],
    )
    def test_check_match_bbox__exceptions(self, bbox_input: Any, match_text: str) -> None:
        """Check that invalid match-bbox input raise proper errors."""

        with pytest.raises(InvalidBoundsError, match=match_text):
            _check_match_bbox(self.rast, bbox_input)

    @pytest.mark.parametrize(
        "points_input, expected_pts, expected_tag_scalar",
        [
            # Tuple of numbers (scalar)
            ((5, 6), (np.array([5.0]), np.array([6.0])), True),
            # Tuple of 1D arrays
            ((np.arange(0, 3), np.arange(3, 6)), (np.arange(0, 3), np.arange(3, 6)), False),
            # Using x/y arrays
            ((np.array([1, 2, 3]), np.array([4, 5, 6])), (np.array([1, 2, 3]), np.array([4, 5, 6])), False),
            # Vector-like (point cloud)
            (
                gu.Vector(gpd.GeoDataFrame({"geometry": [Point(0, 1), Point(1, 2)]}, crs="EPSG:4326")),
                (np.array([0, 1]), np.array([1, 2])),
                False,
            ),
        ],
    )
    def test_check_match_points__valid(self, points_input: Any, expected_pts: Any, expected_tag_scalar: bool) -> None:
        """Check that valid match-points input pass."""

        # Create synthetic raster for source_obj
        transform = rio.transform.from_bounds(0, 0, 10, 10, 5, 5)
        rast = gu.Raster.from_array(np.zeros((5, 5)), transform=transform, crs=4326)

        pts, input_scalar = _check_match_points(rast, points_input)

        # Scalars converted correctly
        assert np.array_equal(pts[0], expected_pts[0])
        assert np.array_equal(pts[1], expected_pts[1])
        # Scalar flag
        assert input_scalar == expected_tag_scalar

        # For vector-like input
        if isinstance(points_input, gu.Vector):
            assert pts[0].shape[0] == len(points_input.gdf)  # type: ignore
            assert pts[1].shape[0] == len(points_input.gdf)  # type: ignore

    @pytest.mark.parametrize(
        "points_input, match_text",
        [
            # Wrong length
            (([1, 2, 3],), "Expected a sequence.*of two array-like objects"),
            (([1, 2, 3], [4, 5], [6]), "Expected a sequence.*of two array-like objects"),
            # Non-numeric
            ((["a", "b"], [1, 2]), "must be numeric"),
            # Mismatched lengths
            (([1, 2], [1, 2, 3]), "must have the same length"),
            # Mismatched dimensions
            ((np.ones((4, 4)), np.ones((4, 4))), "got dimensions"),
            # Invalid object
            (42, "Cannot interpret point input"),
            ("abc", "Cannot interpret point input"),
            (
                gu.Vector(gpd.GeoDataFrame(geometry=[Polygon([(0, 0), (1, 0), (1, 1)])], crs=4326)),
                "must contain only points",
            ),
        ],
    )
    def test_check_match_points__exceptions(self, points_input: Any, match_text: str) -> None:
        """Check that invalid match-points input raise proper errors."""

        transform = rio.transform.from_bounds(0, 0, 10, 10, 5, 5)
        rast = gu.Raster.from_array(np.zeros((5, 5)), transform=transform, crs=4326)

        with pytest.raises(InvalidPointsError, match=match_text):
            _check_match_points(rast, points_input)

    @pytest.mark.parametrize(
        # Source, Reference, Resolution, Shape, Bounds, Coordinates
        # Source = Object from which the match is called (used as fallback for bounds/CRS)
        # Reference = Object passed to match
        "src, ref, res, shape, bounds, coords",
        [
            # 1/ First category: No source fallback (= first column does not matter, it is ignored)
            # Only coords (regular grid)
            ("rast", None, None, None, None, (np.array([0, 1, 2]), np.array([0, 1, 2]))),
            # Bounds + resolution
            ("rast", None, 1, None, (0, 0, 3, 3), None),
            # Bounds + shape
            ("rast", None, None, (3, 3), (0, 0, 3, 3), None),
            # Reference raster
            ("rast", "rast", None, None, None, None),
            # Reference vector + resolution
            ("rast", "vect", 1, None, None, None),
            # Reference vector + shape
            ("rast", "vect", None, (3, 3), None, None),
            # Reference object exposing only bbox and CRS + resolution
            ("rast", "bbox", 1, None, None, None),
            # 2/ Second category: Source fallback (= first column matters)
            # Source fallback for bounds (raster or vector) + resolution
            ("rast", None, 1, None, None, None),
            ("vect", None, 1, None, None, None),
            # Source fallback for resolution (raster only) + bounds
            ("rast", None, None, None, (0, 0, 3, 3), None),
            # Source fallback for resolution (raster only) + reference vector
            ("rast", "vect", None, None, None, None),
        ],
    )
    def test_check_match_grid__valid(self, src: Any, ref: Any, res: Any, shape: Any, bounds: Any, coords: Any) -> None:
        """Check that valid match-grid input pass."""
        # Synthetic raster/vector
        rast = gu.Raster.from_array(np.zeros((5, 5)), transform=rio.transform.from_bounds(0, 0, 10, 10, 5, 5), crs=4326)
        vect = gu.Vector(gpd.GeoDataFrame({"geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]}, crs="EPSG:4326"))

        ref_input = None
        if ref == "rast":
            ref_input = rast
        elif ref == "vect":
            ref_input = vect
        elif ref == "bbox":
            ref_input = SimpleNamespace(bbox=(0, 0, 1, 1), crs=pyproj.CRS.from_epsg(4326))

        if src == "rast":
            src_input = rast
        elif src == "vect":
            src_input = vect

        out_shape, out_transform, out_crs = _check_match_grid(
            src=src_input, ref=ref_input, res=res, shape=shape, bounds=bounds, coords=coords, crs=None
        )

        # Sanity checks
        assert out_crs == src_input.crs
        assert isinstance(out_shape, tuple) and len(out_shape) == 2
        assert isinstance(out_transform, rio.Affine)
        if ref == "bbox":
            assert rio.transform.array_bounds(*out_shape, out_transform) == pytest.approx((0, 0, 1, 1))

    @pytest.mark.parametrize(
        "kwargs, error_match, warn",
        [
            # Invalid reference objects (error only)
            ({"ref": 42}, "Cannot interpret reference grid", None),
            ({"ref": "abcd"}, "Cannot interpret reference grid", None),
            ({"ref": {"bounds": (0, 0, 1, 1)}}, "Cannot interpret reference grid", None),
            # Vector reference: missing res / shape
            ({"src": vect, "ref": vect}, "requires a provided resolution 'res' or grid shape 'shape'", None),
            ({"src": vect, "ref": vect_gdf}, "requires a provided resolution 'res' or grid shape 'shape'", None),
            # Vector reference: both res and shape
            ({"ref": vect, "res": 1, "shape": (10, 10)}, "Both 'res' and 'shape' were passed", None),
            ({"ref": vect_gdf, "res": 1, "shape": (10, 10)}, "Both 'res' and 'shape' were passed", None),
            # Raster ref ignores redundant args (warning only)
            (
                {"ref": rast, "res": 1},
                None,
                (IgnoredGridWarning, "already defines a complete grid"),
            ),
            (
                {"ref": rast_xr, "bounds": (0, 0, 10, 10)},
                None,
                (IgnoredGridWarning, "already defines a complete grid"),
            ),
            # Manual grid: res + shape together (error)
            (
                {"res": 1, "shape": (10, 10), "bounds": (0, 0, 10, 10)},
                "Both output grid resolution 'res' and shape 'shape'",
                None,
            ),
            # Manual grid: coords with other grid definitions
            (
                {"src": vect, "coords": (np.arange(5), np.arange(5)), "res": 1},
                None,
                (IgnoredGridWarning, "already defines a complete grid"),
            ),
            (
                {"src": vect, "coords": (np.arange(5), np.arange(5)), "shape": (10, 10)},
                None,
                (IgnoredGridWarning, "already defines a complete grid"),
            ),
            (
                {"coords": (np.arange(5), np.arange(5)), "bounds": (0, 0, 10, 10)},
                None,
                (IgnoredGridWarning, "already defines a complete grid"),
            ),
            # Insufficient inputs
            ({"src": vect}, "Insufficient inputs to define a complete grid", None),
        ],
    )
    def test_check_match_grid__exceptions(
        self, kwargs: dict[str, Any], error_match: str, warn: tuple[Any, str]
    ) -> None:
        """Check that invalid match-grid input raise proper errors."""

        init_kwargs = {
            "src": None,
            "ref": None,
            "res": None,
            "shape": None,
            "bounds": None,
            "coords": None,
            "crs": None,
        }
        init_kwargs.update(kwargs)
        if init_kwargs["src"] is None:
            init_kwargs["src"] = self.rast

        if warn is not None:
            warn_cls, warn_match = warn
            with pytest.warns(warn_cls, match=warn_match):
                if error_match:
                    with pytest.raises(InvalidGridError, match=error_match):
                        _check_match_grid(**init_kwargs)  # type: ignore
                else:
                    _check_match_grid(**init_kwargs)  # type: ignore
        else:
            if error_match:
                with pytest.raises(InvalidGridError, match=error_match):
                    _check_match_grid(**init_kwargs)
            else:
                _check_match_grid(**init_kwargs)

    @pytest.mark.parametrize("reference", [None, SimpleNamespace(crs=None), SimpleNamespace(crs="invalid-crs")])
    def test_get_reproject_crs__error_invalid_crs(self, reference: Any) -> None:
        """Checks an error is raised for invalid explicit CRSs and missing or invalid reference CRSs."""

        with pytest.raises(InvalidCRSError):
            _get_reproject_crs(ref=reference, crs="invalid-crs" if reference is None else None)
