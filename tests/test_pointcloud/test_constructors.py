"""Test point cloud construction from coordinate and value arrays."""

from typing import Literal

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from geopandas.testing import assert_geodataframe_equal

import geoutils as gu
from geoutils._misc import import_optional


class TestPointCloudConstructors:
    """Test module for default output, representations and auxiliary attributes in from_xyz()."""

    def test_from_xyz__default(self) -> None:
        """Checks that from_xyz() returns a DataArray with the supplied coordinates, values and CRS."""

        # Create three elevations from separate coordinate and value lists
        points = gu.pointcloud.from_xyz([5, 15, 25], [25, 15, 5], [100, 110, 120], crs=32631)

        # Check the default name, point dimension and exact coordinates and elevations
        assert isinstance(points, xr.DataArray)
        assert points.name == "z" and points.dims == ("point",)
        np.testing.assert_array_equal(points.pc.to_array(), [[5, 15, 25], [25, 15, 5], [100, 110, 120]])
        assert points.pc.crs.to_epsg() == 32631
        assert points.pc.is_loaded

    @pytest.mark.parametrize("as_object_type", ["dataarray", "dataset", "geodataframe"])
    @pytest.mark.parametrize("data_name", [None, "height"])
    @pytest.mark.parametrize("use_z", [False, True])
    def test_from_xyz__representations_and_attributes(
        self, as_object_type: Literal["dataarray", "dataset", "geodataframe"], data_name: str | None, use_z: bool
    ) -> None:
        """Checks that every representation preserves point values, CRS and large integer auxiliary attributes."""

        # Create IDs above the exact float64 integer range to expose unwanted dtype conversion
        x = np.array([5.0, 15.0, 25.0])
        y = np.array([25.0, 15.0, 5.0])
        values = np.array([100.0, 110.0, 120.0])
        identifiers = np.array([2**63 + 1, 2**63 + 2, 2**63 + 3], dtype=np.uint64)

        # Construct the requested representation with an auxiliary measurement
        points = gu.pointcloud.from_xyz(
            x,
            y,
            values,
            32631,
            data_name,
            use_z,
            auxiliary={"identifier": identifiers},
            as_object_type=as_object_type,
        )
        name = data_name or "z"
        active = points[name] if as_object_type == "dataset" else points

        # Check ordered values, metadata and auxiliary dtype for each output type
        expected_type = {"dataarray": xr.DataArray, "dataset": xr.Dataset, "geodataframe": gpd.GeoDataFrame}
        assert isinstance(points, expected_type[as_object_type])
        np.testing.assert_array_equal(active.pc.to_array(), np.stack([x, y, values]))
        np.testing.assert_array_equal(points["identifier"], identifiers)
        assert points["identifier"].dtype == np.uint64
        assert active.pc.crs.to_epsg() == 32631 and active.pc.is_loaded
        if as_object_type == "dataset":
            assert list(points.pc.compatible_variables) == [name, "identifier"]
            assert "x_point" in points.coords and "y_point" in points.coords
            assert points.identifier.pc.crs == active.pc.crs
        elif as_object_type == "geodataframe":
            assert np.all(points.geometry.has_z == use_z)
            assert points.pc.data_name == (None if use_z else name)


class TestPointCloudConstructorsChunked:
    """Test module for lazy values, coordinates and auxiliary arrays in every from_xyz() representation."""

    @pytest.mark.parametrize("as_object_type", ["dataarray", "dataset", "geodataframe"])
    @pytest.mark.parametrize("lazy_field", ["values", "coordinates", "auxiliary"])
    @pytest.mark.parametrize("use_z", [False, True])
    def test_from_xyz__loading_laziness(
        self, as_object_type: Literal["dataarray", "dataset", "geodataframe"], lazy_field: str, use_z: bool
    ) -> None:
        """Checks that constructing from lazy arrays runs no tasks and exactly matches eager construction."""
        import_optional("dask")
        import dask.array as da
        from dask.callbacks import Callback

        if as_object_type == "geodataframe":
            import_optional("dask_geopandas")

        # Split seven points into chunks of 3/3/1 to reveal missing or duplicated final rows
        x, y, values = np.arange(7.0), np.arange(7.0) + 10, np.arange(7.0) + 100
        identifiers = np.arange(7, dtype=np.uint64) + np.uint64(2**63)
        arrays = {"x": x, "y": y, "z": values, "identifier": identifiers}
        lazy_names = {"values": ["z"], "coordinates": ["x", "y"], "auxiliary": ["identifier"]}[lazy_field]
        for name in lazy_names:
            arrays[name] = da.from_array(arrays[name], chunks=3)
            assert arrays[name].chunks == ((3, 3, 1),)

        # Construct both outputs, forbidding task execution during lazy construction
        eager = gu.pointcloud.from_xyz(
            x,
            y,
            values,
            32631,
            use_z=use_z,
            auxiliary={"identifier": identifiers},
            as_object_type=as_object_type,
        )
        with Callback(pretask=lambda *args: pytest.fail("Constructing points must not compute input arrays")):
            lazy = gu.pointcloud.from_xyz(
                arrays["x"],
                arrays["y"],
                arrays["z"],
                32631,
                use_z=use_z,
                auxiliary={"identifier": arrays["identifier"]},
                as_object_type=as_object_type,
            )

        # Check which point values are loaded before computing the output
        active = lazy["z"] if as_object_type == "dataset" else lazy
        active_is_loaded = as_object_type == "dataset" and lazy_field == "auxiliary"
        assert active.pc.is_loaded == active_is_loaded
        actual = lazy.compute()
        if as_object_type == "geodataframe":
            assert_geodataframe_equal(actual, eager)
        else:
            xr.testing.assert_identical(actual, eager)
        assert (actual["z"] if as_object_type == "dataset" else actual).pc.is_loaded

        # Check computing the output leaves the input graphs and lazy result unchanged
        assert active.pc.is_loaded == active_is_loaded
        for name in lazy_names:
            assert isinstance(arrays[name], da.Array)
        if as_object_type != "geodataframe":
            if lazy_field == "values":
                variable = active
            elif lazy_field == "coordinates":
                variable = active.pc.ds["x"]
            else:
                variable = lazy["identifier"]
            assert variable.chunks


class TestPointCloudConstructorsErrors:
    """Test module for invalid output types and point attributes in from_xyz()."""

    def test_from_xyz__error_invalid_type(self) -> None:
        """Checks an error is raised for an unsupported point cloud representation."""

        # Request a representation outside the public choices
        with pytest.raises(ValueError, match="as_object_type"):
            gu.pointcloud.from_xyz([0], [0], [1], 32631, as_object_type="scene")  # type: ignore[arg-type]

    @pytest.mark.parametrize("as_object_type", ["dataarray", "dataset", "geodataframe"])
    def test_from_xyz__error_reserved_attribute(
        self, as_object_type: Literal["dataarray", "dataset", "geodataframe"]
    ) -> None:
        """Checks an error is raised when an auxiliary attribute has a coordinate name."""

        # Try replacing X coordinates with an auxiliary measurement
        with pytest.raises(ValueError, match="distinct"):
            gu.pointcloud.from_xyz([0], [0], [1], 32631, auxiliary={"x": [2]}, as_object_type=as_object_type)
