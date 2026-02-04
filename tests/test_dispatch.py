"""
Test the dispatch functions used for checking and normalizing inputs.
"""

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import pytest
import rasterio as rio
from shapely.geometry import Point, Polygon

import geoutils as gu
from geoutils._dispatch import _check_coords  # Level-0 checks (manual input)
from geoutils._dispatch import _grid_from_src  # Helpers for grid conversion
from geoutils._dispatch import (  # Level-1 checks (match reference object)
    _check_bounds,
    _check_crs,
    _check_match_bbox,
    _check_match_grid,
    _check_match_points,
    _check_resolution,
    _check_shape,
    _grid_from_bounds_res,
    _grid_from_bounds_shape,
    _grid_from_coords,
)
from geoutils.exceptions import (
    IgnoredGridWarning,
    InvalidBoundsError,
    InvalidCRSError,
    InvalidGridError,
    InvalidPointsError,
    InvalidResolutionError,
    InvalidShapeError,
)


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
        bbox = pd.DataFrame(data={"minx": 0, "miny": 1, "maxx": 2, "maxy": 3}, index=[0])
        xmin, ymin, xmax, ymax = _check_bounds(bbox)
        assert (xmin, ymin, xmax, ymax) == (0, 1, 2, 3)

        # Valid dataframe (bounding box of geodataframe)
        bbox = {"left": 0, "bottom": 1, "right": 2, "top": 3}
        xmin, ymin, xmax, ymax = _check_bounds(bbox)
        assert (xmin, ymin, xmax, ymax) == (0, 1, 2, 3)

    @pytest.mark.parametrize(
        "bbox, match_text",
        [
            ([0, 1, 2], "length of 3"),  # Invalid length
            ([5, 0, 2, 4], "xmin must be < xmax"),  # Invalid ordering, xmin >= xmax
            ([0, 5, 4, 2], "ymin must be < ymax"),  # Invalid ordering, ymin >= ymax
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
        """Check that invalid bounds input raise proper errors."""

        with pytest.raises(InvalidBoundsError, match=match_text):
            _check_bounds(bbox)  # type: ignore

    def test_check_resolution__valid(self) -> None:
        """Check that valid resolution input pass."""

        # Valid scalar resolution, normalized to tuple
        assert _check_resolution(5) == (5, 5)

        # Valid sequence resolution, normalized to tuple
        assert _check_resolution([1.0, 2.0]) == (1.0, 2.0)  # type: ignore

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

    @pytest.mark.parametrize(
        "shape, match_text",
        [
            (10, "sequence of two integers"),  # Not a sequence
            ((10,), "length 1"),  # Wrong length
            ((-1, 5), "non-negative"),  # Negative
            (("a", 5), "must be integers"),  # Non-integer
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
        ],
    )
    def test_check_coords__exceptions(self, coords: Any, match_text: str) -> None:
        """Check that invalid coordinate inputs raise proper errors."""
        with pytest.raises(InvalidGridError, match=match_text):
            _check_coords(coords)


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

    # Raster and vector classes
    rast = gu.Raster.from_array(np.zeros((5, 5)), transform=rio.transform.from_bounds(0, 0, 10, 10, 5, 5), crs=4326)
    rast2 = gu.Raster.from_array(np.zeros((5, 5)), transform=rio.transform.from_bounds(0, 0, 5, 5, 5, 5), crs=4326)
    vect = gu.Vector(gpd.GeoDataFrame({"geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])]}, crs="EPSG:4326"))
    vect2 = gu.Vector(gpd.GeoDataFrame({"geometry": [Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])]}, crs="EPSG:4326"))
    # Xarray DataArray and Vector geodataframe
    rast_xr = rast.to_xarray()
    vect_gdf = vect.ds

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
            assert pts[0].shape[0] == len(points_input.ds)  # type: ignore
            assert pts[1].shape[0] == len(points_input.ds)  # type: ignore

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
