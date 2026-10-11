"""Tests for referencing helper functions for points (dimensions, X/Y coordinates and CRS)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

from geoutils.pointcloud.referencing import _point_coordinates, _point_crs
from tests.accessor_helpers import mixed_dataset as mixed_dataset


class TestPointReferencing:
    """Test module for ``_point_coordinates()`` and ``_point_crs()`` internal functions."""

    @pytest.mark.parametrize("representation", ["dataarray", "dataset"])
    @pytest.mark.parametrize(
        "dimension,x_name,y_name",
        [("point", "x", "y"), ("point", "x_point", "y_point"), ("observation", "x_observation", "y_observation")],
    )
    def test_point_coordinates__names(self, representation: str, dimension: str, x_name: str, y_name: str) -> None:
        """Checks that X/Y names resolve the common point dimension in DataArrays and Datasets."""

        # We create coordinates that share one dimension
        coordinates = {x_name: (dimension, [0.0, 1.0, 2.0]), y_name: (dimension, [3.0, 4.0, 5.0])}
        values = xr.DataArray([10.0, 20.0, 30.0], dims=dimension, coords=coordinates, name="height")
        points = values if representation == "dataarray" else values.to_dataset()
        original = points.copy(deep=True)

        # Check dimension/names
        assert _point_coordinates(points) == (dimension, x_name, y_name)
        xr.testing.assert_identical(points, original)

    @pytest.mark.parametrize(
        "attribute,x_value,y_value",
        [
            ("axis", "X", "Y"),
            ("standard_name", "projection_x_coordinate", "projection_y_coordinate"),
            ("standard_name", "longitude", "latitude"),
        ],
    )
    def test_point_coordinates__cf_attributes(self, attribute: str, x_value: str, y_value: str) -> None:
        """Checks that CF attributes identify point coordinates with otherwise unrecognized names."""

        # We use CF metadata for coordinates with arbitrary names
        points = xr.DataArray(
            [10.0, 20.0], dims="observation", coords={"east": ("observation", [0, 1]), "north": ("observation", [2, 3])}
        )
        points.east.attrs[attribute] = x_value
        points.north.attrs[attribute] = y_value

        # Check both coordinate axes
        assert _point_coordinates(points) == ("observation", "east", "north")
        assert points.east.attrs[attribute] == x_value
        assert points.north.attrs[attribute] == y_value

    def test_point_coordinates__mixed_dataset(self, mixed_dataset: xr.Dataset) -> None:
        """Checks that point variables are resolved independently of raster/scalar variables with other dimensions."""

        # We use the mixed dataset fixture, and store point locations as data variables beside a raster independent
        # X/Y dimensions
        points = mixed_dataset.reset_coords(["x_point", "y_point"])
        original = points.copy(deep=True)

        # Check point coordinates
        assert _point_coordinates(points) == ("point", "x_point", "y_point")
        assert "x_point" in points.data_vars and "y_point" in points.data_vars
        xr.testing.assert_identical(points, original)

    @pytest.mark.parametrize("metadata", ["values", "x", "y", "all"])
    def test_point_crs__attributes(self, metadata: str) -> None:
        """Checks that value and coordinate CRS attributes resolve to the same PyProj reference."""

        # Integer EPSG codes, strings and WKT can all describe the same point support
        reference = CRS.from_epsg(32631)
        points = xr.DataArray([10, 20], dims="point", coords={"x": ("point", [0, 1]), "y": ("point", [2, 3])})
        if metadata in ("values", "all"):
            points.attrs["crs"] = 32631
        if metadata in ("x", "all"):
            points.x.attrs["crs"] = "EPSG:32631"
        if metadata in ("y", "all"):
            points.y.attrs["crs"] = reference.to_wkt(version="WKT1_GDAL")

        # Check all resolve to the same reference CRS
        assert _point_crs(points) == reference

    @pytest.mark.parametrize("value_crs", [None, 32631])
    @pytest.mark.parametrize("location", ["attrs", "encoding"])
    @pytest.mark.parametrize("description", ["crs_wkt", "spatial_ref", "cf"])
    def test_point_crs__grid_mapping(self, location: str, description: str, value_crs: int | None) -> None:
        """Checks that an explicit grid mapping provides the CRS through WKT/CF attributes."""

        # Create grid mapping manually
        # (CF-only metadata omits WKT so PyProj must reconstruct the projection from its parameters)
        reference = CRS.from_epsg(32631)
        if description == "cf":
            metadata: dict[str, Any] = reference.to_cf()
            metadata.pop("crs_wkt")
        else:
            metadata = {description: reference.to_wkt()}
        mapping = xr.Variable((), 0, attrs=metadata)
        points = xr.DataArray(
            [10, 20],
            dims="point",
            coords={"x": ("point", [0, 1]), "y": ("point", [2, 3]), "point_ref": mapping},
        )
        getattr(points, location)["grid_mapping"] = "point_ref"
        if value_crs is not None:
            points.attrs["crs"] = value_crs

        # Check mapping describes the point support
        # (CF-only metadata can omit the original geographic axis order)
        result = _point_crs(points)
        assert result is not None and result.equals(reference, ignore_axis_order=True)

    def test_point_crs__independent_raster_mapping(self, mixed_dataset: xr.Dataset) -> None:
        """Checks that an independent raster grid mapping is ignored when resolving point CRS metadata."""

        # A scalar raster mapping accompanies extracted point values in a mixed Dataset
        points = mixed_dataset.point_z.copy(deep=True)
        points.attrs.pop("crs")
        assert "spatial_ref" in points.coords

        # Only point metadata may supply the CRS, even when the raster uses another projection
        assert _point_crs(points) is None
        points.attrs["crs"] = "EPSG:4326"
        assert _point_crs(points) == CRS.from_epsg(4326)


class TestPointReferencingChunked:
    """Test module for ``_point_coordinates()`` and ``_point_crs()`` for lazy input."""

    @pytest.mark.parametrize("representation", ["dataarray", "dataset"])
    def test_methods__loading_laziness(self, representation: str) -> None:
        """Checks that coordinate and CRS resolution run no tasks and match the same calls on eager point data."""

        pytest.importorskip("dask")
        from dask.callbacks import Callback

        # Create + write chunked data array with point coords
        eager = xr.DataArray(
            np.arange(7.0),
            dims="point",
            coords={"x": ("point", np.arange(7.0)), "y": ("point", np.zeros(7))},
            attrs={"crs": "EPSG:32631"},
            name="height",
        )
        lazy = eager.chunk({"point": 3})
        source = lazy if representation == "dataarray" else lazy.to_dataset()
        reference = eager if representation == "dataarray" else eager.to_dataset()
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(1)):
            coordinates = _point_coordinates(source)
            crs = _point_crs(lazy)

        # Check metadata matches eager inputs, with lazy graphs unloaded
        assert not tasks
        assert coordinates == _point_coordinates(reference)
        assert crs == _point_crs(eager)
        assert lazy.chunks and lazy.x.chunks and lazy.y.chunks
        xr.testing.assert_identical(source.compute(), reference)
        assert lazy.chunks and lazy.x.chunks and lazy.y.chunks


class TestPointReferencingErrors:
    """Test module for errors/warning in point referencing functions."""

    @pytest.mark.parametrize("names", [[], ["x"], ["y"]])
    def test_point_coordinates__error_missing_pair(self, names: list[str]) -> None:
        """Checks an error is raised when one or both point coordinate axes are missing."""

        # Create coords with missing axes
        coordinates = {name: ("point", [0, 1]) for name in names}
        points = xr.DataArray([10, 20], dims="point", coords=coordinates)
        with pytest.raises(AttributeError, match="require x/y coordinates"):
            _point_coordinates(points)

    def test_point_coordinates__error_different_dimensions(self) -> None:
        """Checks an error is raised when X and Y describe different dimensions."""

        # Create points with X/Y on different dims
        points = xr.Dataset({"height": ("point", [10, 20]), "x": ("point", [0, 1]), "y": ("other", [2, 3])})
        with pytest.raises(AttributeError, match="require x/y coordinates"):
            _point_coordinates(points)

    @pytest.mark.parametrize("coordinates", ["duplicate_axis", "independent_support"])
    def test_point_coordinates__error_ambiguous_pair(self, coordinates: str) -> None:
        """Checks an error is raised for competing X coordinates or two independent point supports."""

        # Create ambiguous support
        points = xr.Dataset({"height": ("point", [10, 20]), "x": ("point", [0, 1]), "y": ("point", [2, 3])})
        if coordinates == "duplicate_axis":
            points["x_point"] = ("point", [4, 5])
        else:
            points["x_other"] = ("other", [4, 5])
            points["y_other"] = ("other", [6, 7])
        with pytest.raises(ValueError, match="ambiguous"):
            _point_coordinates(points)

    def test_point_coordinates__error_multiple_dimensions(self) -> None:
        """Checks an error is raised for a DataArray with an additional value dimension."""

        # Give a wrong 2D shape for a 1D input
        points = xr.DataArray(
            np.ones((2, 2)), dims=("point", "band"), coords={"x": ("point", [0, 1]), "y": ("point", [2, 3])}
        )
        with pytest.raises(AttributeError, match="one-dimensional DataArray"):
            _point_coordinates(points)

    @pytest.mark.parametrize("values", [["a", "b"], [False, True]])
    def test_point_coordinates__error_nonnumeric(self, values: list[str] | list[bool]) -> None:
        """Checks an error is raised for text or boolean point coordinates."""

        # Create invalid DataArray with string or boolean values
        points = xr.DataArray([10, 20], dims="point", coords={"x": ("point", values), "y": ("point", [2, 3])})
        with pytest.raises(AttributeError, match="must be numeric"):
            _point_coordinates(points)

    @pytest.mark.parametrize("axis", ["x", "y"])
    def test_point_crs__error_coordinate_crs(self, axis: str) -> None:
        """Checks an error is raised when a coordinate CRS disagrees with the point values."""

        points = xr.DataArray(
            [10, 20], dims="point", coords={"x": ("point", [0, 1]), "y": ("point", [2, 3])}, attrs={"crs": 32631}
        )
        points.coords[axis].attrs["crs"] = 4326
        with pytest.raises(ValueError, match="values and X/Y coordinates must have the same CRS"):
            _point_crs(points)

    def test_point_crs__error_grid_mapping_crs(self) -> None:
        """Checks an error is raised when the explicitly referenced grid mapping disagrees with point values."""

        # We use a mapping that contradicts the CRS
        mapping = xr.Variable((), 0, attrs={"spatial_ref": CRS.from_epsg(4326).to_wkt()})
        points = xr.DataArray(
            [10, 20],
            dims="point",
            coords={"x": ("point", [0, 1]), "y": ("point", [2, 3]), "point_ref": mapping},
            attrs={"crs": 32631},
        )
        points.encoding["grid_mapping"] = "point_ref"

        with pytest.raises(ValueError, match="values and their grid mapping must have the same CRS"):
            _point_crs(points)
