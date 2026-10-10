"""Tests for Xarray functions shared by point cloud and raster accessors, mostly for Dataset selection/rebuilding."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

from geoutils._misc import import_optional
from geoutils._xarray import _rebuild_dataset, _same_coordinate, _select_dataset_variables
from tests.accessor_helpers import mixed_dataset as mixed_dataset


class TestSelectDatasetVariables:
    """Test module for shared ``_select_dataset_variables()`` function that identifies compatible."""

    @pytest.mark.parametrize("variables", [None, ["slope"], ["slope", "dem"]])
    def test_select_dataset_variables__selection(self, mixed_dataset: xr.Dataset, variables: list[str] | None) -> None:
        """Checks that default selection uses compatible variables and explicit selection preserves its order."""

        # Raster and point values coexist, but this selection operates on the two raster values
        compatible = ["dem", "slope"]
        original = mixed_dataset.copy(deep=True)
        expected = compatible.copy() if variables is None else variables.copy()

        # Return a separate list without changing the available names or input Dataset
        selected = _select_dataset_variables(mixed_dataset, compatible, variables)
        assert selected == expected
        assert selected is not compatible and selected is not variables
        assert compatible == ["dem", "slope"]
        xr.testing.assert_identical(mixed_dataset, original)


class TestSameCoordinate:
    """Test module for shared ``same_coordinate`` function."""

    @pytest.mark.parametrize(
        "dimension,values,units,expected",
        [
            ("point", [1.0, np.nan], "m", True),
            ("point", [1.0, 0.0], "m", False),
            ("observation", [1.0, np.nan], "m", False),
            ("point", [1.0], "m", False),
            ("point", [1.0, np.nan], "km", False),
        ],
    )
    def test_same_coordinate__values_and_metadata(
        self, dimension: str, values: list[float], units: str, expected: bool
    ) -> None:
        """Checks that coordinate equality requires matching dimensions, values and units, including missing values."""

        # Equal numbers with different dimensions or units cannot identify the same locations
        original = xr.Variable("point", [1.0, np.nan], attrs={"units": "m"})
        other = xr.Variable(dimension, values, attrs={"units": units})

        # Equality treats matching missing values as equal and distinguishes spatial metadata
        assert _same_coordinate(original, other) is expected
        assert _same_coordinate(other, original) is expected

    @pytest.mark.parametrize("description", ["wkt", "cf"])
    def test_same_coordinate__equivalent_grid_mapping(self, description: str) -> None:
        """Checks that equivalent WKT and CF descriptions with numerically equal transforms identify the same grid."""

        # Two valid spellings describe the same projection and pixel grid
        reference = CRS.from_epsg(32631)
        original = xr.Variable(
            (),
            0,
            attrs={
                "spatial_ref": reference.to_wkt(version="WKT1_GDAL"),
                "GeoTransform": "0 1 0 2 0 -1",
                "label": "grid",
            },
        )
        if description == "cf":
            # NetCDF mappings may store projection parameters without a WKT description
            original.attrs.update(reference.to_cf())
            original.attrs.pop("crs_wkt")
            original.attrs.pop("spatial_ref")
        attributes = {**reference.to_cf(), "GeoTransform": "0.0 1.0 0.0 2.0 0.0 -1.0", "label": "grid"}
        other = xr.Variable((), 0, attrs=attributes)

        # CRS and transform equality are independent of WKT version or numeric formatting
        assert _same_coordinate(original, other)
        assert _same_coordinate(other, original)

    @pytest.mark.parametrize("mapping", [False, True])
    @pytest.mark.parametrize("changed", [False, True])
    def test_same_coordinate__array_attributes(self, mapping: bool, changed: bool) -> None:
        """Checks that coordinate and grid mapping comparison accepts equal arrays and detects changed attributes."""

        # NetCDF valid ranges and custom mapping metadata commonly contain arrays rather than scalars
        attributes = {"valid_range": np.array([0.0, 7.0])}
        original = (
            xr.Variable((), 0, attrs=attributes) if mapping else xr.Variable("point", [0.0, 1.0], attrs=attributes)
        )
        other = original.copy(deep=True)
        if mapping:
            reference = CRS.from_epsg(32631)
            original.attrs["spatial_ref"] = reference.to_wkt(version="WKT1_GDAL")
            other.attrs.update(reference.to_cf())
        if changed:
            other.attrs["valid_range"][1] = 8.0

        # Equivalent CRS spellings must not hide different custom array attributes
        assert _same_coordinate(original, other) is not changed
        assert _same_coordinate(other, original) is not changed

    @pytest.mark.parametrize("change", ["crs", "transform", "custom_attribute"])
    def test_same_coordinate__different_grid_mapping(self, change: str) -> None:
        """Checks that a changed CRS, affine transform or custom grid attribute makes mappings unequal."""

        # Every field describes spatial meaning or metadata that reconstruction must preserve
        attributes = {"spatial_ref": CRS.from_epsg(32631).to_wkt(), "GeoTransform": "0 1 0 2 0 -1", "label": "grid"}
        original = xr.Variable((), 0, attrs=attributes)
        other = original.copy(deep=True)
        if change == "crs":
            other.attrs["spatial_ref"] = CRS.from_epsg(32632).to_wkt()
        elif change == "transform":
            other.attrs["GeoTransform"] = "1 1 0 2 0 -1"
        else:
            other.attrs["label"] = "another grid"

        # A coordinate replacement must not discard a meaningful metadata difference
        assert not _same_coordinate(original, other)
        assert not _same_coordinate(other, original)


class TestRebuildDataset:
    """Test module for ``_rebuild_dataset`` function."""

    def test_rebuild_dataset__values_and_metadata(self, mixed_dataset: xr.Dataset) -> None:
        """Checks that replacing values on an unchanged grid preserves unrelated variables, indexes and metadata."""

        # Replace only DEM values, including its output encoding and the Dataset's source metadata
        source = mixed_dataset.copy(deep=True)
        source.encoding = {"source": "survey"}
        dem = source.dem.copy(data=source.dem.data + 10)
        dem.encoding = {"grid_mapping": "spatial_ref", "dtype": "float32"}
        expected = source.copy(deep=True)
        expected["dem"] = dem

        # The complete Dataset must equal a native replacement on the original coordinates
        result = _rebuild_dataset(source, {"dem": dem})
        xr.testing.assert_identical(result, expected)
        assert result.xindexes["x"] is source.xindexes["x"]
        assert result.xindexes["y"] is source.xindexes["y"]
        assert result.encoding == source.encoding and result.encoding is not source.encoding
        assert result.dem.encoding == dem.encoding
        assert result.attrs == source.attrs and result.attrs is not source.attrs
        np.testing.assert_array_equal(source.dem.data, mixed_dataset.dem.data)

    @pytest.mark.parametrize("representation", ["raster", "pointcloud"])
    def test_rebuild_dataset__selected_coordinates(self, mixed_dataset: xr.Dataset, representation: str) -> None:
        """Checks that replacing shared spatial coordinates selects exact rows without aligning onto old coordinates."""

        # Select a smaller grid or repeated point rows while leaving the other representation independent
        if representation == "raster":
            names, dimension, positions = ["dem", "slope"], "x", [0, 2, 4]
        else:
            names, dimension, positions = ["point_z", "point_sigma"], "point", [1, 3, 3, 8]
        transformed = {name: mixed_dataset[name].isel({dimension: positions}) for name in names}
        expected = mixed_dataset.isel({dimension: positions})

        # Native positional selection is the reference for values, labels and independent variables
        result = _rebuild_dataset(mixed_dataset, transformed)
        xr.testing.assert_identical(result, expected)
        assert result.sizes[dimension] == len(positions)
        assert mixed_dataset.sizes[dimension] > result.sizes[dimension]

    def test_rebuild_dataset__value_coordinate_role(self, mixed_dataset: xr.Dataset) -> None:
        """Checks that an existing value carried as an auxiliary output coordinate remains a Dataset data variable."""

        # Point algorithms can attach other value fields to their active DataArray
        points = mixed_dataset.point_z.copy(data=mixed_dataset.point_z.data * 2)
        points = points.assign_coords(point_sigma=mixed_dataset.point_sigma.variable)
        expected = mixed_dataset.copy(deep=True)
        expected.point_z.data *= 2

        # Reconstruction must not promote the independent value to a coordinate or drop it
        result = _rebuild_dataset(mixed_dataset, {"point_z": points})
        xr.testing.assert_identical(result, expected)
        assert "point_sigma" in result.data_vars and "point_sigma" not in result.coords

    def test_rebuild_dataset__equivalent_grid_mapping(self, mixed_dataset: xr.Dataset) -> None:
        """Checks that equivalent WKT metadata permits an untouched raster on the same grid."""

        # Rewrite only WKT spelling, preserving the projection, transform and all other attributes
        mapping = mixed_dataset.spatial_ref.variable.copy(deep=True)
        wkt = CRS.from_epsg(32631).to_wkt(version="WKT1_GDAL")
        mapping.attrs.update(crs_wkt=wkt, spatial_ref=wkt)
        dem = mixed_dataset.dem.assign_coords(spatial_ref=mapping)

        # Existing coordinate metadata remains valid for both selected and untouched raster values
        result = _rebuild_dataset(mixed_dataset, {"dem": dem})
        xr.testing.assert_identical(result, mixed_dataset)

    def test_rebuild_dataset__equivalent_point_crs(self, mixed_dataset: xr.Dataset) -> None:
        """Checks that equivalent point CRS descriptions preserve unselected values on the same locations."""

        # An EPSG code and the stored WKT describe the same point support
        points = mixed_dataset.point_z.copy(deep=True)
        points.attrs["crs"] = 32631

        # Equivalent CRS metadata changes only the selected variable's description
        result = _rebuild_dataset(mixed_dataset, {"point_z": points})
        xr.testing.assert_identical(result.point_z, points)
        xr.testing.assert_identical(result.point_sigma, mixed_dataset.point_sigma)
        xr.testing.assert_identical(result.dem, mixed_dataset.dem)

    def test_rebuild_dataset__independent_time_dimension(self, mixed_dataset: xr.Dataset) -> None:
        """Checks that a changed raster CRS leaves independent values on the raster's time dimension valid."""

        # Temperature uses the same dates as two raster slices but has no spatial dimensions
        dates = np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[ns]")
        source = mixed_dataset.assign(
            dem=mixed_dataset.dem.expand_dims(time=dates),
            slope=mixed_dataset.slope.expand_dims(time=dates),
            temperature=xr.DataArray([0.0, 1.0], dims="time", coords={"time": dates}),
        )
        transformed = {name: source[name].rio.write_crs(32632) for name in ("dem", "slope")}

        # Changing spatial meaning affects X/Y, while time-only values and labels remain unchanged
        result = _rebuild_dataset(source, transformed)
        xr.testing.assert_identical(result.temperature.variable, source.temperature.variable)
        xr.testing.assert_identical(result.time.variable, source.time.variable)
        assert result.xindexes["time"] is source.xindexes["time"]
        xr.testing.assert_identical(result.point_z.variable, source.point_z.variable)
        for name in transformed:
            assert result[name].rio.crs.to_epsg() == 32632


class TestXarrayChunked:
    """Test module for coordinate comparison and Dataset reconstruction without implicitly computing Dask arrays."""

    @pytest.mark.parametrize("changed", [False, True])
    def test_same_coordinate__loading_laziness(self, changed: bool) -> None:
        """Checks that lazy coordinate comparison runs no tasks and matches eager equality."""

        import_optional("dask")
        import dask.array as da
        from dask.callbacks import Callback

        # Chunks of 3/3/1 rows include a shorter final coordinate block
        coordinate = xr.Variable("point", da.from_array(np.arange(7.0), chunks=3), attrs={"units": "m"})
        other = coordinate.copy(deep=False)
        if changed:
            other = xr.Variable("point", coordinate.data + 1, attrs=coordinate.attrs)
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(1)):
            result = _same_coordinate(coordinate, other)

        # Only explicit computation may read coordinate values
        assert not tasks
        assert coordinate.chunks and other.chunks
        assert result == _same_coordinate(coordinate.compute(), other.compute())
        assert result is not changed
        assert coordinate.chunks and other.chunks

    def test_rebuild_dataset__loading_laziness(self, mixed_dataset: xr.Dataset) -> None:
        """Checks that reconstruction preserves lazy point labels and independent raster graphs until computation."""

        import_optional("dask")
        import dask.array as da
        from dask.callbacks import Callback

        # Stored labels use unindexed Dask coordinates so construction must not create a Pandas index
        source = mixed_dataset.chunk({"point": 4, "x": 4, "y": 3})
        labels = xr.Coordinates({"point": xr.Variable("point", da.arange(9, chunks=4) * 10)}, indexes={})
        source = source.assign_coords(labels)
        positions = [1, 3, 3, 8]
        transformed = {name: source[name].isel(point=positions) for name in ("point_z", "point_sigma")}
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(1)):
            result = _rebuild_dataset(source, transformed)

        # Repeated selected rows remain lazy, and unrelated raster values reuse their original graph
        assert not tasks
        assert source.point.chunks and source.point_z.chunks
        assert result.point.chunks and result.point_z.chunks
        assert "point" not in source.xindexes and "point" not in result.xindexes
        assert result.dem.data is source.dem.data
        assert result.xindexes["x"] is source.xindexes["x"]

        # Computing the complete output matches reconstruction from the same eager selection
        eager = source.compute()
        eager_transformed = {name: eager[name].isel(point=positions) for name in transformed}
        expected = _rebuild_dataset(eager, eager_transformed)
        xr.testing.assert_identical(result.compute(), expected)
        assert source.point.chunks and source.point_z.chunks and source.dem.chunks


class TestXarrayErrors:
    """Test module for invalid variable selection and coordinate changes incompatible with untouched Dataset values."""

    @pytest.mark.parametrize(
        "variables,message",
        [
            ([], "nonempty"),
            ("dem", "distinct"),
            (["dem", "dem"], "distinct"),
            (["missing"], "Unknown"),
            (["x"], "Unknown"),
            (["point_z"], "incompatible"),
        ],
    )
    def test_select_dataset_variables__error_selection(
        self, mixed_dataset: xr.Dataset, variables: str | list[str], message: str
    ) -> None:
        """Checks an error is raised for empty, repeated, unknown, coordinate or incompatible value selections."""

        # Only DEM and slope are compatible value variables for this selection
        with pytest.raises(ValueError, match=message):
            _select_dataset_variables(mixed_dataset, ["dem", "slope"], variables)

    def test_select_dataset_variables__error_empty_representation(self, mixed_dataset: xr.Dataset) -> None:
        """Checks an error is raised when default selection finds no compatible variables."""

        # A Dataset can contain values without any compatible values for the requested representation
        with pytest.raises(ValueError, match="nonempty"):
            _select_dataset_variables(mixed_dataset, [], None)

    @pytest.mark.parametrize("representation", ["raster", "pointcloud"])
    @pytest.mark.parametrize("change", ["coordinates", "size", "crs"])
    def test_rebuild_dataset__error_untouched_spatial_values(
        self, mixed_dataset: xr.Dataset, representation: str, change: str
    ) -> None:
        """Checks an error is raised when unselected raster or point values share changed coordinates, sizes or CRS."""

        # Select one value while leaving another value on the same support untouched
        if representation == "raster":
            name, untouched, dimension, coordinate = "dem", "slope", "x", "x"
        else:
            name, untouched, dimension, coordinate = "point_z", "point_sigma", "point", "x_point"
        value = mixed_dataset[name]
        if change == "coordinates":
            transformed = value.assign_coords({coordinate: value.coords[coordinate] + 1})
        elif change == "size":
            transformed = value.isel({dimension: [0, 1]})
        elif representation == "raster":
            transformed = value.rio.write_crs(4326)
        else:
            transformed = value.copy(deep=True)
            transformed.attrs["crs"] = 4326
        original = mixed_dataset.copy(deep=True)

        # Untouched values cannot silently acquire new locations or a different reference system
        with pytest.raises(ValueError, match=f"Untouched variables.*{untouched}"):
            _rebuild_dataset(mixed_dataset, {name: transformed})
        xr.testing.assert_identical(mixed_dataset, original)

    def test_rebuild_dataset__error_conflicting_results(self, mixed_dataset: xr.Dataset) -> None:
        """Checks an error is raised when two selected rasters produce different coordinates on their common axis."""

        # Each result is individually valid but the two X axes cannot describe one Dataset dimension
        transformed = {
            "dem": mixed_dataset.dem,
            "slope": mixed_dataset.slope.assign_coords(x=mixed_dataset.x + 1),
        }

        # Coordinate conflicts must be reported before Xarray can align or pad the value arrays
        with pytest.raises(ValueError, match="conflicting coordinates for 'x'"):
            _rebuild_dataset(mixed_dataset, transformed)

    def test_rebuild_dataset__error_value_coordinate_conflict(self, mixed_dataset: xr.Dataset) -> None:
        """Checks an error is raised when an output coordinate conflicts with an existing Dataset value variable."""

        # An auxiliary point field differs from the untouched value stored under the same name
        points = mixed_dataset.point_z.assign_coords(point_sigma=mixed_dataset.point_sigma + 1)

        # Reconstruction cannot choose between two meanings for the existing value field
        with pytest.raises(ValueError, match="coordinate 'point_sigma' conflicts with an existing value variable"):
            _rebuild_dataset(mixed_dataset, {"point_z": points})

    def test_rebuild_dataset__error_missing_auxiliary_coordinate(self, mixed_dataset: xr.Dataset) -> None:
        """Checks an error is raised when a changed point dimension has no replacement for a dependent coordinate."""

        # Both point values select fewer rows but deliberately omit a coordinate tied to those rows
        source = mixed_dataset.assign_coords(quality=("point", np.arange(9)))
        transformed = {
            name: source[name].isel(point=[1, 3, 8]).drop_vars("quality") for name in ("point_z", "point_sigma")
        }

        # Old auxiliary coordinates cannot be attached to shortened or reordered point values
        with pytest.raises(ValueError, match="Coordinate 'quality'.*no transformed replacement"):
            _rebuild_dataset(source, transformed)

    def test_rebuild_dataset__error_untouched_grid_mapping(self, mixed_dataset: xr.Dataset) -> None:
        """Checks an error is raised for a scalar explicitly tied to a changed CRS through its grid mapping."""

        # A distance threshold uses the raster CRS even though it has no spatial dimensions
        source = mixed_dataset.assign(distance_threshold=xr.DataArray(5.0, attrs={"units": "m"}))
        source.distance_threshold.encoding["grid_mapping"] = "spatial_ref"
        transformed = {name: source[name].rio.write_crs(4326) for name in ("dem", "slope")}

        # Replacing the referenced mapping cannot leave its dependent scalar metadata unchanged
        with pytest.raises(ValueError, match="Untouched variables.*distance_threshold"):
            _rebuild_dataset(source, transformed)

    def test_rebuild_dataset__error_untouched_point_grid_mapping(self, mixed_dataset: xr.Dataset) -> None:
        """Checks an error is raised when changing a point grid mapping would invalidate unselected point values."""

        # One explicitly referenced mapping supplies the CRS for the common point support
        mapping = xr.Variable((), 0, attrs={"crs_wkt": CRS.from_epsg(32631).to_wkt()})
        source = mixed_dataset.copy(deep=True).assign_coords(point_ref=mapping)
        for name in ("point_z", "point_sigma"):
            source[name].attrs.pop("crs")
        source.point_z.encoding["grid_mapping"] = "point_ref"
        new_mapping = xr.Variable((), 0, attrs={"crs_wkt": CRS.from_epsg(4326).to_wkt()})
        points = source.point_z.assign_coords(point_ref=new_mapping)

        # The changed point reference invalidates shared rows even without raster spatial dimensions
        with pytest.raises(ValueError, match="Untouched variables.*point_sigma"):
            _rebuild_dataset(source, {"point_z": points})
