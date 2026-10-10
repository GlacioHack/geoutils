"""Tests module for the ``pc`` Xarray accessors."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import CRS
from shapely.geometry import box

import geoutils as gu
from geoutils._dispatch import _get_pointcloud_interface, _is_pointcloud, _is_raster, is_dask_array
from geoutils._misc import import_optional
from geoutils.pointcloud.base import GeometryPointCloudBase, PointCloudBase
from tests.accessor_helpers import (
    assert_dataset_output_equal,
    assert_variograms_equal,
    assert_xarray_equal,
)
from tests.accessor_helpers import (
    mixed_dataset as mixed_dataset,
)
from tests.test_pointcloud.test_pd_accessor import TestPointCloudAccessor as PointCloudAccessorReference


@pytest.fixture
def pointcloud_file(tmp_path: Any) -> Any:
    """Fixture for tests, providing a GeoParquet writer to compare lazy point operations with eager results"""

    import geopandas as gpd

    pytest.importorskip("pyarrow")
    filenames = []

    def write_points(frame: gpd.GeoDataFrame, partitions: int) -> tuple[str, int, str | None]:
        """Save a point fixture and return its filename, chunk size and data name."""

        # Copy the dataframe and select its active values
        frame = frame.copy()
        column = frame.pc.data_name
        if "data_name" not in frame.attrs and column is None:
            column = next((name for name in ("height", "z", "intensity") if name in frame), None)
        if column is None and not frame.geometry.has_z.all():
            # Add zero elevations when neither the attributes nor the geometry provide point values
            frame["z"] = np.zeros(len(frame))
            column = "z"

        # Write rows by chunk
        chunks = max(1, int(np.ceil(len(frame) / partitions)))
        filename = tmp_path / f"points-{len(filenames)}.parquet"
        filenames.append(filename)
        frame.to_parquet(
            filename, index=True, geometry_encoding="geoarrow", schema_version="1.1.0", row_group_size=chunks
        )
        return str(filename), chunks, column

    return write_points


class TestDataArrayPointCloudAccessor:
    """Test module for the Xarray ``pc`` accessor."""

    # Reuse the point locations and values from the Pandas accessor tests
    arr_points = PointCloudAccessorReference.arr_points
    gdf = PointCloudAccessorReference.gdf
    fn_las = PointCloudAccessorReference.fn_las

    def test_accessor(self) -> None:
        """Checks that point cloud metadata, values and conversion are exposed through the accessor."""

        # We convert the shared GeoDataFrame to a Xarray point cloud with z as the active values
        ds = gu.PointCloud(self.gdf, data_name="z").to_xarray()

        # Compare the accessor view with the underlying array
        assert ds.pc.data_name == "z"
        assert ds.pc.point_count == len(ds)
        np.testing.assert_array_equal(ds.pc.data, ds.data)
        assert isinstance(ds.pc.to_geoutils(), gu.PointCloud)

    def test_open_pointcloud(self, tmp_path: Path) -> None:
        """Checks that a point file opens as a DataArray per default."""

        # We write the point fixture to a point file
        filename = tmp_path / "test.gpkg"
        self.gdf.to_file(filename)

        # Then open it eagerly and check its type, loading state and exact point cloud equality
        ds = gu.open_pointcloud(filename, data_name="z")
        assert isinstance(ds, xr.DataArray)
        assert ds.pc.is_loaded
        assert ds.pc.to_geoutils().pointcloud_equal(gu.PointCloud(self.gdf, data_name="z"))

    def test_from_xyz(self) -> None:
        """Checks that DataArray construction from X/Y/Z arrays."""

        # We create a DataArray from X/Y/Z arrays
        ds = gu.DataArrayPointCloudAccessor.from_xyz(
            x=self.arr_points[:, 0], y=self.arr_points[:, 1], z=self.arr_points[:, 2], crs=4326, data_name="z"
        )

        # Check DataArray type and exact equality
        assert isinstance(ds, xr.DataArray)
        assert ds.pc.to_geoutils().pointcloud_equal(gu.PointCloud(self.gdf, data_name="z"))

    def test_from_xyz__values_coords_attrs(self) -> None:
        """Checks that separate arrays preserve dtypes and attributes."""

        # Create float32 values and uint64 IDs above float64 precision (to reveal accidental dtype conversion)
        values = np.array([1, 2, 3], dtype=np.float32)
        identifiers = np.array([2**63 + 1, 2**63 + 2, 2**63 + 3], dtype=np.uint64)

        # We construct points with the IDs as auxiliary attributes
        points = gu.DataArrayPointCloudAccessor.from_xyz(
            [0, 1, 2],
            [3, 4, 5],
            values,
            32633,
            data_name="height",
            auxiliary={"id": identifiers},
        )

        # Check public metadata and point arrays
        assert points.pc.data_name == "height"
        assert points.pc.crs == CRS.from_epsg(32633)
        assert points.pc.point_count == 3
        assert points.pc.bounds == (0, 3, 2, 5)
        assert points.pc.is_loaded
        np.testing.assert_array_equal(points.pc.data, values)
        np.testing.assert_array_equal(points.coords["id"], identifiers)
        assert points.dtype == np.float32
        assert points.coords["id"].dtype == np.uint64

    @pytest.mark.parametrize("construction", ["from_array", "from_tuples"])
    def test_from_array(self, construction: str) -> None:
        """Checks that from_arrays and from_tuples construct the same point values."""

        # Create X/Y/Z arrays
        coordinates = np.arange(15).reshape(3, 5)
        expected = gu.DataArrayPointCloudAccessor.from_xyz(coordinates[0], coordinates[1], coordinates[2], crs=32633)

        # Construct points through array/tuple
        if construction == "from_array":
            result = gu.DataArrayPointCloudAccessor.from_array(coordinates.T, crs=32633)
        else:
            result = gu.DataArrayPointCloudAccessor.from_tuples(coordinates.T.tolist(), crs=32633)

        # Check exact equality
        xr.testing.assert_identical(result, expected)
        np.testing.assert_array_equal(result.pc.to_array(), coordinates)
        assert result.pc.to_tuples() == [tuple(row) for row in coordinates.T]

    def test_accessor__inheritance_and_dispatch(self) -> None:
        """Checks point classes/accessors inherit from PointCloudBase and DataArrays dispatches to accessor."""

        # We create each point representation from the same X/Y/Z arrays
        points = gu.DataArrayPointCloudAccessor.from_xyz([0, 1], [2, 3], [4, 5], 32633)
        cloud = gu.PointCloud.from_xyz([0, 1], [2, 3], [4, 5], 32633)
        frame = cloud.gdf

        # Check inheritance and distinguish 1D arrays from rasters
        assert isinstance(cloud, GeometryPointCloudBase)
        assert isinstance(frame.pc, gu.GeoPandasPointCloudAccessor)
        assert isinstance(points.pc, gu.DataArrayPointCloudAccessor)
        assert all(isinstance(interface, PointCloudBase) for interface in (cloud, frame.pc, points.pc))
        assert _is_pointcloud(points)
        assert not _is_raster(points)
        assert _get_pointcloud_interface(points) is points.pc

    def test_copy(self) -> None:
        """Checks the copy function."""

        # We create points with auxiliary IDs, then copy them with replacement values
        points = gu.DataArrayPointCloudAccessor.from_xyz([0, 1], [2, 3], [4, 5], 32633, auxiliary={"id": [6, 7]})
        copied = points.pc.copy(new_array=np.array([8, 9]))

        # We then moodify the metadata and IDs (to check that the source is independent, no back-propag)
        copied.attrs["label"] = "copy"
        copied.coords["id"][0] = 10

        # Check equality of arrays and metadata, and no back propag
        np.testing.assert_array_equal(points.data, [4, 5])
        np.testing.assert_array_equal(copied.data, [8, 9])
        np.testing.assert_array_equal(points.coords["id"], [6, 7])
        assert "label" not in points.attrs

    def test_set_data_name__auxiliary(self) -> None:
        """Checks setting an auxiliary data as main data."""

        # We create points with float elevations, and uint8 classification
        points = gu.DataArrayPointCloudAccessor.from_xyz(
            [0, 1],
            [2, 3],
            [4.0, 5.0],
            32633,
            auxiliary={"class": np.array([6, 7], dtype=np.uint8)},
        )

        # We set auxiliary values as main, and check both source and output
        selected = points.pc.set_data_name("class")
        np.testing.assert_array_equal(selected.data, [6, 7])
        np.testing.assert_array_equal(selected.coords["z"], [4, 5])
        assert selected.dtype == np.uint8
        assert selected.pc.data_name == "class"
        assert points.pc.data_name == "z"

    @pytest.mark.parametrize("lazy", [False, True])
    def test_to_geoutils__loading_laziness(self, tmp_path: Path, lazy: bool) -> None:
        """Checks that to_geoutils() loads exact values, while keep source lazy (if Dask)."""

        if lazy:
            import_optional("dask")

        # We write points with float32 values and uint64 IDs in groups of 3/3/1
        reference = gu.PointCloud.from_xyz(np.arange(7), np.arange(7), np.arange(7, dtype=np.float32), 32633)
        reference.gdf["id"] = np.arange(7, dtype=np.uint64) + 2**63
        filename = tmp_path / "conversion.parquet"
        reference.to_parquet(filename, chunks=3)

        # Then open all point attributes (eagerly or in chunks)
        source = gu.open_pointcloud(filename, columns="all", chunks=3 if lazy else None)
        graph = source.data
        assert source.pc.is_loaded is not lazy

        # We run the conversion to a PointCloud, and check the source Dask array stays lazy
        result = source.pc.to_geoutils()
        assert isinstance(result, gu.PointCloud) and result.is_loaded
        assert source.data is graph
        assert source.pc.is_loaded is not lazy

        # And finally check exact equality
        assert reference.pointcloud_equal(result)

    def test_cross_type_outputs_are_accessors(self) -> None:
        """Checks that a Xarray accessor is returned when gridding changes the geospatial data type."""

        # We create points covering a small regular raster grid
        ds = gu.DataArrayPointCloudAccessor.from_xyz(
            x=np.array([0, 1, 0, 1]), y=np.array([0, 0, 1, 1]), z=np.array([1, 2, 3, 4]), crs=3857, data_name="z"
        )

        # We grid the points
        raster = ds.pc.grid(
            grid_coords=(np.array([0, 1]), np.array([0, 1])), resampling="nearest", dist_nodata_pixel=10
        )

        # And check raster output is a DataArray (that converts to a Raster)
        assert isinstance(raster, xr.DataArray)
        assert isinstance(raster.rst.to_geoutils(), gu.Raster)

    @pytest.mark.parametrize("points", [xr.DataArray([1, 2]), xr.DataArray(np.ones((2, 2)), dims=("y", "x"))])
    def test_accessor__error_invalid_coordinates(self, points: xr.DataArray) -> None:
        """Checks an error is raised for arrays without one X/Y coordinate per point row."""

        with pytest.raises(RuntimeError, match="initializing"):
            _ = points.pc

    def test_from_xyz__error_mismatched_lengths(self) -> None:
        """Checks an error is raised for coordinate/value arrays with different lengths."""

        with pytest.raises(ValueError, match="same length"):
            gu.DataArrayPointCloudAccessor.from_xyz([0, 1], [2], [3, 4], 32633)

    def test_from_xyz__error_reserved_attribute(self) -> None:
        """Checks an error is raised for an auxiliary attribute that overwrites a point coordinate."""

        with pytest.raises(ValueError, match="distinct"):
            gu.DataArrayPointCloudAccessor.from_xyz([0], [1], [2], 32633, auxiliary={"x": [3]})

    @pytest.mark.parametrize("options", [{}, {"bbox": (0.0, 0.0, 1.0, 1.0), "mode": "invalid"}])
    def test_crop__error_invalid_bounds_or_mode(self, options: dict[str, Any]) -> None:
        """Checks an error is raised for missing bounds or an unsupported crop mode."""

        # We create valid points, and the error comes from crop arguments
        points = gu.DataArrayPointCloudAccessor.from_xyz([0.0], [1.0], [2.0], 32633)
        with pytest.raises(ValueError, match="Argument '(bbox|mode)'"):
            points.pc.crop(**options)


class TestDatasetPointCloudAccessor:
    """Test module for the Xarray Dataset point cloud accessor ``pc``."""

    def test_accessor(self, mixed_dataset: xr.Dataset) -> None:
        """
        First, we check that point and raster accessors can "coexist" and discover their respective variables.

        We test variable discovery separately because a mixed Dataset needs to distinguish its point values from its
        rasters.
        """

        assert isinstance(mixed_dataset.pc, gu.DatasetPointCloudAccessor)
        assert not isinstance(mixed_dataset.pc, PointCloudBase)
        assert isinstance(mixed_dataset.point_z.pc, gu.DataArrayPointCloudAccessor)
        assert mixed_dataset.pc.variables == ["point_z", "point_sigma"]
        assert not hasattr(mixed_dataset.pc, "ds")

        # Check Dataset dispatch and no access to spatial metadata at Dataset level (only DataArray)
        assert _get_pointcloud_interface(mixed_dataset) is None
        assert not hasattr(mixed_dataset.pc, "crs")
        assert not hasattr(mixed_dataset.pc, "bounds")

    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize(
        "method,options,variables",
        [
            ("reproject", {"crs": 32631}, None),
            ("reproject", {"crs": 32631}, ["point_z"]),
            ("reproject", {"crs": 4326}, None),
            ("filter", {"method": "mean", "radius": 2}, None),
            ("filter", {"method": "mean", "radius": 2}, ["point_z"]),
        ],
    )
    def test_methods__point_outputs(
        self, mixed_dataset: xr.Dataset, method: str, options: dict[str, Any], variables: list[str] | None, lazy: bool
    ) -> None:
        """
        Checks that Dataset methods producing point cloud outputs match DataArray ops per recognized point variable,
        and do not affect independent rasters and scalar variables in the dataset.

        We test methods with/without selecting a specific variable.

        These combinations cover point outputs with basic arguments. Other types of outputs (rasters, dict),
        and changed Dataset behaviour for input arguments are tested separately further below.
        """

        # 1/ Use the mixed Xarray Dataset from the mixed_dataset fixture
        # Use the current CRS when leaving point values unselected
        source = mixed_dataset
        if lazy:
            pytest.importorskip("dask")
            from dask.callbacks import Callback

            # We chunk the mixed dataset into 4/4/1 point rows (to test partial chunks)
            source = source.chunk({"point": 4, "x": 4, "y": 3})
        original = source.copy(deep=True)

        # 2/ We run the same operation through Dataset and DataArray point cloud accessors
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(1)) if lazy else nullcontext():
            result = getattr(source.pc, method)(variables=variables, **options)
        names = source.pc.variables if variables is None else variables
        if lazy:
            # Check that construction is lazy and independent raster values are unchanged
            assert not tasks
            assert all(source[name].chunks and result[name].chunks for name in source.pc.variables)
            assert result.dem.data is source.dem.data
            for name in source.pc.variables:
                if name not in names:
                    assert result[name].data is source[name].data
        outputs = {}
        for name in names:
            # Run the DataArray method and restore the Dataset point coordinate names
            expected = getattr(mixed_dataset[name].pc, method)(**options).rename(name)
            outputs[name] = expected.rename({"x": "x_point", "y": "y_point"})
            assert isinstance(result[name].pc, gu.DataArrayPointCloudAccessor)

        # 3/ Check exact equality
        assert_dataset_output_equal(source, result.compute() if lazy else result, outputs)
        xr.testing.assert_identical(source, original)
        if lazy:
            assert all(source[name].chunks for name in source.pc.variables)

    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize(
        "method,options,variables",
        [
            ("grid", {"resampling": "nearest"}, None),
            ("grid", {"resampling": "nearest"}, ["point_sigma", "point_z"]),
            ("grid", {"resampling": "nearest"}, ["point_z"]),
            ("grid", {"bounds": (0, 0, 7, 5), "res": 1, "resampling": "nearest"}, None),
        ],
    )
    def test_methods__raster_outputs(
        self, mixed_dataset: xr.Dataset, method: str, options: dict[str, Any], variables: list[str] | None, lazy: bool
    ) -> None:
        """
        Checks that Dataset methods producing raster outputs match DataArray ops per recognized point variable, and
        do not affect independent rasters and scalar variables in the dataset.

        We test methods with a raster reference or explicit grid bounds, selecting one or several point variables.
        """

        # 1/ Use the mixed Xarray Dataset from the mixed_dataset fixture
        # Reuse the raster grid so independent raster values remain valid
        source = mixed_dataset
        reference_options = options.copy()
        if "bounds" not in options:
            reference_options["ref"] = source.dem
        if lazy:
            pytest.importorskip("dask")
            from dask.callbacks import Callback

            # Chunk the mixed dataset into 4/4/1 point rows (to test partial chunks)
            source = source.chunk({"point": 4, "x": 4, "y": 3})
        options = options.copy()
        if "bounds" not in options:
            # Use the independent DEM as the same grid reference for eager and chunked calls
            options["ref"] = source.dem
        original = source.copy(deep=True)

        # 2/ We run the same operation through Dataset and DataArray point cloud accessors
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(1)) if lazy else nullcontext():
            result = getattr(source.pc, method)(variables=variables, **options)
        names = source.pc.variables if variables is None else variables
        if lazy:
            # Check that construction is lazy and independent raster values are unchanged
            assert not tasks
            assert all(source[name].chunks and result[name].chunks for name in names)
            assert result.dem.data is source.dem.data
            for name in source.pc.variables:
                if name not in names:
                    assert result[name].data is source[name].data
        outputs = {}
        for name in names:
            # Run the DataArray method on the same reference grid or explicit bounds
            outputs[name] = getattr(mixed_dataset[name].pc, method)(**reference_options).rename(name)
            assert isinstance(result[name].rst, gu.DataArrayRasterAccessor)

        # 3/ Check exact equality
        assert_dataset_output_equal(source, result.compute() if lazy else result, outputs)
        xr.testing.assert_identical(source, original)
        if lazy:
            assert all(source[name].chunks for name in source.pc.variables)
            assert source.x_point.chunks and source.y_point.chunks

    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize(
        "method,options,with_mask,compare_outputs",
        [
            pytest.param(
                "variogram",
                {
                    "n_pairs": 12,
                    "strategy": "kdtree",
                    "anchors_per_round": 32,
                    "n_runs": 2,
                    "bins": [0, 2, 4, 8],
                    "min_lag": 0.1,
                    "max_lag": 8,
                    "estimator": np.mean,
                    "random_state": 7,
                },
                True,
                assert_variograms_equal,
                id="variogram",
            ),
        ],
    )
    @pytest.mark.parametrize(
        "variables,coordinates,random_state",
        [
            (None, "names", "seed"),
            (["point_sigma"], "cf", "seed"),
            (["point_sigma", "point_z"], "variables", "generator"),
        ],
    )
    def test_methods__dict_outputs(
        self,
        mixed_dataset: xr.Dataset,
        method: str,
        options: dict[str, Any],
        with_mask: bool,
        compare_outputs: Callable[[dict[str, Any], dict[str, Any]], None],
        variables: list[str] | None,
        coordinates: str,
        random_state: str,
        lazy: bool,
    ) -> None:
        """
        Checks that Dataset methods returning dictionaries match separate DataArray ops for the selected point
        variables, and do not change the source dataset.

        We test variable selection, native/CF point coordinates and X/Y data variables, with shared masks, bins and
        random states.
        """

        # 1/ Create a test mixed Xarray Dataset by modifying the mixed_dataset fixture
        # We copy the mixed dataset and add NaNs at different rows
        source = mixed_dataset.copy(deep=True)
        source.point_z.data[[1, 6]] = np.nan
        source.point_sigma.data[[0, 4]] = np.nan

        # Rename X/Y with CF axis attributes when requested
        if coordinates == "cf":
            source = source.rename({"x_point": "easting", "y_point": "northing"})
            source.easting.attrs["axis"] = "X"
            source.northing.attrs["axis"] = "Y"

        # Copy the method options and optionally exclude row 2 with the same mask for both calculations
        names = source.pc.variables if variables is None else variables
        options = options.copy()
        if with_mask:
            options["mask"] = np.arange(9) != 2

        # Save the eager source for independent DataArray calculations
        reference = source

        # Optionally convert X/Y to data variables
        if coordinates == "variables":
            source = source.reset_coords(["x_point", "y_point"])
        if lazy:
            from unittest.mock import Mock

            pytest.importorskip("dask")
            from dask import array as dask_array
            from dask import delayed
            from dask.callbacks import Callback

            # Split points into chunks of 4/4/1 (to test partial chunks)
            source = source.chunk({"point": 4, "x": 4, "y": 3})
            read_unused = Mock(side_effect=AssertionError("Variograms must not read unselected point values."))
            unused = dask_array.from_delayed(
                delayed(read_unused)(), shape=source.point_z.shape, dtype=source.point_z.dtype
            )
            source = source.assign(unused=source.point_z.copy(data=unused))
            variables = names
        original = source.copy(deep=True)

        # 2/ Run Dataset and DataArray methods with the same mask, bins and random state
        # Run the DataArray method per variable name, using a seed or one shared random generator
        reference_options = options.copy()
        if "random_state" in reference_options:
            reference_options["random_state"] = 7 if random_state == "seed" else np.random.default_rng(7)
        expected = {name: getattr(reference[name].pc, method)(**reference_options) for name in names}

        # We reset the random state and pass bin edges as an iterator
        # The Dataset call must reuse those edges and process variables in the same order as the reference
        if "random_state" in options:
            options["random_state"] = 7 if random_state == "seed" else np.random.default_rng(7)
        if "bins" in options and not isinstance(options["bins"], str):
            options["bins"] = iter(options["bins"])

        # Run the Dataset method while recording any reads of chunked inputs
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(1)) if lazy else nullcontext():
            result = getattr(source.pc, method)(variables=variables, **options)
        if lazy:
            # Check that selected tasks ran without reading the unused field or changing the lazy source arrays
            assert tasks
            assert not read_unused.called
            assert all(source[name].chunks for name in source.pc.variables)
            assert source.dem.chunks
            if method == "variogram":
                assert all(isinstance(variogram.semivariance, np.ndarray) for variogram in result.values())

        # 3/ Check exact equality
        assert isinstance(result, dict)
        assert list(result) == names
        compare_outputs(result, expected)
        xr.testing.assert_identical(source, original)

    @pytest.mark.parametrize("lazy", [False, True])
    @pytest.mark.parametrize(
        "method,options,coordinates",
        [
            ("subsample", {"subsample": 1}, "names"),
            ("subsample", {"subsample": 4}, "cf"),
            ("subsample", {"subsample": 0.5}, "variables"),
            ("subsample", {"subsample": 4, "mask": np.arange(9) > 2}, "names"),
            ("crop", {"bbox": (0, 0, 3, 4)}, "names"),
            ("clip", {"mask": box(0, 0, 3, 4)}, "variables"),
        ],
    )
    def test_methods__shared_selection(
        self, mixed_dataset: xr.Dataset, method: str, options: dict[str, Any], coordinates: str, lazy: bool
    ) -> None:
        """
        Checks that Dataset point selections use the same rows as a DataArray ops for every point variable, and do not
        affect independent rasters and scalar variables in the dataset.

        We test native/CF point coordinates and X/Y data variables, with duplicate row labels and mixed value types.
        """

        # 1/ Create a test mixed Xarray Dataset by modifying the mixed_dataset fixture
        # We add categorical fields, integers above 2**63 and duplicate labels to test selection by row position
        source = mixed_dataset.assign(
            classification=("point", np.array(["ground", "tree", "other"] * 3)),
            identifier=("point", np.arange(9, dtype=np.uint64) + 2**63),
        ).assign_coords(point=np.arange(9) // 2)

        # Add missing uncertainty values and optionally rename X/Y with CF axis attributes
        source.point_sigma.data[::2] = np.nan
        if coordinates == "cf":
            source = source.rename({"x_point": "easting", "y_point": "northing"})
            source.easting.attrs["axis"] = "X"
            source.northing.attrs["axis"] = "Y"
        options = dict(options, random_state=3) if method == "subsample" else options

        # Use positional labels on the DataArray to identify the shared row selection independently
        reference = source.point_z.assign_coords(point=np.arange(9))
        reference_options = options.copy()
        if coordinates == "variables":
            source = source.reset_coords(["x_point", "y_point"])
        if lazy:
            pytest.importorskip("dask")
            from dask.callbacks import Callback

            # Split point values into 4/4/1 (to test the shorter final chunk)
            chunks = {"point": 4, "x": 4, "y": 3}
            eager_source = source
            source = source.chunk(chunks)
            if method in ("crop", "clip"):
                # Supply loaded X/Y for crop() and clip() so row selection can determine the output size
                if coordinates == "cf":
                    source = source.assign_coords(easting=eager_source.easting, northing=eager_source.northing)
                elif coordinates == "variables":
                    source = source.assign(x_point=eager_source.x_point, y_point=eager_source.y_point)
                else:
                    source = source.assign_coords(x_point=eager_source.x_point, y_point=eager_source.y_point)
            if method == "subsample" and "mask" in options:
                # Wrap the loaded mask with the lazy point coordinates to select rows without reading measurements
                options = dict(options, mask=source.point_z.copy(data=options["mask"]))
        original = source.copy(deep=True)

        # 2/ Run Dataset and DataArray point selections with the same options
        positions = getattr(reference.pc, method)(**reference_options).point.data
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(1)) if lazy else nullcontext():
            result = getattr(source.pc, method)(**options)
        if lazy:
            # Check that construction is lazy and independent raster values are unchanged
            assert not tasks
            assert all(source[name].chunks and result[name].chunks for name in source.pc.variables)
            assert result.dem.data is source.dem.data

        # 3/ Compare values and metadata
        expected = source.isel(point=positions)
        if coordinates == "variables":
            # Mark X/Y as coordinates to match how the Dataset accessor returns them
            expected = expected.set_coords(["x_point", "y_point"])
        assert_xarray_equal(result.compute() if lazy else result, expected.compute() if lazy else expected)
        xr.testing.assert_identical(source, original)
        if lazy:
            assert all(source[name].chunks for name in source.pc.variables)

            # Subsampling can leave X/Y lazy because its row count is already known
            if method == "subsample":
                x_name, y_name = ("easting", "northing") if coordinates == "cf" else ("x_point", "y_point")
                assert source[x_name].chunks and source[y_name].chunks

    @pytest.mark.parametrize("mask_form", ["array", "dataarray", "unlabelled_dataarray", "vector"])
    def test_subsample__mask(self, mixed_dataset: xr.Dataset, mask_form: str) -> None:
        """
        Checks mask for Dataset subsampling.

        We need this additional test for accepting array/DataArray/vector masks when the first point variable is
        entirely missing.
        """

        import geopandas as gpd

        # We copy the mixed Dataset fixture, add row labels and replace all elevations with NaN
        source = mixed_dataset.copy(deep=True).assign_coords(point=np.arange(9) * 10)
        source.point_z.data[:] = np.nan
        original = source.copy(deep=True)

        # Then, we create equivalent mask types for all input points
        eligible = (source.x_point.data < 3) & (source.y_point.data < 4)
        masks = {
            "array": eligible,
            "dataarray": source.point_z.copy(data=eligible),
            "unlabelled_dataarray": xr.DataArray(eligible, dims="point"),
            "vector": gu.Vector(gpd.GeoDataFrame(geometry=[box(0, 0, 3, 4)], crs=32631)),
        }

        # Run subsample() for all points
        result = source.pc.subsample(1, mask=masks[mask_form], random_state=3)
        expected = source.isel(point=np.flatnonzero(eligible))

        # Check exact equality
        assert_xarray_equal(result, expected)
        assert result.sizes["point"] == 4
        xr.testing.assert_identical(source, original)

    @pytest.mark.parametrize("coordinates", ["names", "cf"])
    def test_reproject__shared_coords(self, mixed_dataset: xr.Dataset, coordinates: str) -> None:
        """
        Checks that Dataset reprojection matches DataArray point coordinates, and does not affect independent raster
        georeferencing.

        We need this additional test for updating shared native/CF X/Y coordinates and units while preserving custom
        attributes during reprojection.
        """

        # We prepare the mixed dataset with native/CF point coordinates
        source = mixed_dataset
        x_name, y_name = "x_point", "y_point"
        if coordinates == "cf":
            # Rename X/Y and add CF axes, units and custom attributes to check reprojection updates
            source = source.rename({x_name: "east", y_name: "north"})
            x_name, y_name = "east", "north"
            source[x_name].attrs = {"axis": "X", "units": "m", "sensor": "lidar", "valid_range": [0.0, 7.0]}
            source[y_name].attrs = {"axis": "Y", "units": "m"}
        original = source.copy(deep=True)

        # We reproject through Dataset and DataArray point cloud interfaces
        expected = source.point_z.pc.reproject(crs=4326)
        result = source.pc.reproject(crs=4326)

        # Check exact values, CRS and independent raster georeferencing
        np.testing.assert_array_equal(result[x_name], expected.x)
        np.testing.assert_array_equal(result[y_name], expected.y)
        np.testing.assert_array_equal(result.point_z, mixed_dataset.point_z)
        np.testing.assert_array_equal(result.point_sigma, mixed_dataset.point_sigma)
        assert result.point_z.pc.crs.to_epsg() == 4326
        assert result.dem.rst.crs.to_epsg() == 32631
        xr.testing.assert_identical(result.dem, mixed_dataset.dem)

        # Check that reprojected X/Y have geographic CF names and units, with the custom sensor attribute preserved
        assert result[x_name].attrs["standard_name"] == "longitude"
        assert result[y_name].attrs["standard_name"] == "latitude"
        assert result[x_name].attrs["units"] == "degrees_east"
        assert result[y_name].attrs["units"] == "degrees_north"
        if coordinates == "cf":
            assert result[x_name].attrs["sensor"] == "lidar"
            assert "valid_range" not in result[x_name].attrs
        xr.testing.assert_identical(source, original)

    @pytest.mark.parametrize("attribute", ["point_crs", "crs"])
    def test_reproject__dataset_crs(self, mixed_dataset: xr.Dataset, attribute: str) -> None:
        """
        Checks that point CRS falls back to Dataset attributes and point_crs updates after reprojection.

        We need this additional test for reading the point CRS from Dataset attributes when point variables have no CRS
        metadata.
        """

        # We copy the mixed dataset and describe the point CRS only in a Dataset attribute
        source = mixed_dataset.copy(deep=True)
        source.attrs[attribute] = 32631
        for name in source.pc.variables:
            source[name].attrs.pop("crs")
        original = source.copy(deep=True)

        # Reproject using the Dataset CRS and compare coordinates with the georeferenced DataArray call
        expected = mixed_dataset.point_z.pc.reproject(crs=4326)
        result = source.pc.reproject(crs=4326)
        np.testing.assert_array_equal(result.x_point, expected.x)
        np.testing.assert_array_equal(result.y_point, expected.y)

        # Check the point CRS, unchanged raster georeferencing and updated Dataset-level point_crs
        assert all(result[name].pc.crs.to_epsg() == 4326 for name in result.pc.variables)
        if attribute == "point_crs":
            assert CRS.from_user_input(result.attrs[attribute]).to_epsg() == 4326
        xr.testing.assert_identical(result.dem, mixed_dataset.dem)
        xr.testing.assert_identical(source, original)

    @pytest.mark.parametrize("description", ["wkt", "cf", "cf_and_epsg"])
    def test_reproject__point_grid_mapping(self, mixed_dataset: xr.Dataset, description: str) -> None:
        """
        Checks that Dataset reprojection updates the point grid mapping without modifying the independent raster
        mapping.

        We need this additional test for reading WKT/CF/EPSG descriptions and updating a point grid mapping during
        reprojection.
        """

        # We add a separate point grid mapping alongside the fixture's raster mapping
        source = mixed_dataset.assign_coords(point_ref=0)
        crs = source.point_z.pc.crs
        assert crs is not None
        source.point_ref.attrs = CRS.from_epsg(crs.to_epsg()).to_cf()
        if description != "wkt":
            # Remove WKT to test CF projection fields, which can omit the EPSG/WKT geographic axis order
            source.point_ref.attrs.pop("crs_wkt")

        # Link point variables to the new mapping instead of their CRS attributes
        for name in source.pc.variables:
            source[name].attrs.pop("crs")
            source[name].encoding["grid_mapping"] = "point_ref"
        if description == "cf_and_epsg":
            # Describe uncertainty values with EPSG metadata to test equivalent CRS descriptions on shared points
            source.point_sigma.encoding.pop("grid_mapping")
            source.point_sigma.attrs["crs"] = crs.to_epsg()
        original = source.copy(deep=True)

        # We reproject Dataset points to geographic coordinates
        result = source.pc.reproject(crs=4326)

        # Check point CRS updates and unchanged raster metadata
        for name in source.pc.variables:
            assert result[name].pc.crs.to_epsg() == 4326
            assert result[name].encoding.get("grid_mapping") == source[name].encoding.get("grid_mapping")
        xr.testing.assert_identical(result.spatial_ref.variable, source.spatial_ref.variable)
        xr.testing.assert_identical(source, original)

    def test_filter__shared_neighbors(self, mixed_dataset: xr.Dataset, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Checks that filter() on a Dataset calculates neighbors only once and matches separate DataArray calls.

        We need this test to check that all selected point variables share one neighbor search.
        """

        import geoutils.filters.irregular as irregular

        # We count calls to the neighbor search to check that the Dataset method calculates neighbors only once
        # (All nine points fit in one neighborhood batch)
        query = irregular._query_point_neighbours
        queries = []

        def count_query(*args: Any, **kwargs: Any) -> Any:
            """Record each neighborhood query before calculating its point pairs."""
            queries.append(1)
            return query(*args, **kwargs)

        # We filter every DataArray first, then run Dataset filtering with the neighbor counter installed
        expected = {name: mixed_dataset[name].pc.filter("mean", radius=2) for name in mixed_dataset.pc.variables}
        monkeypatch.setattr(irregular, "_query_point_neighbours", count_query)
        result = mixed_dataset.pc.filter("mean", radius=2)

        # Check one neighbor search, exact values and the unchanged DEM
        assert len(queries) == 1
        for name, reference in expected.items():
            np.testing.assert_array_equal(result[name], reference)
        xr.testing.assert_identical(result.dem, mixed_dataset.dem)

    @pytest.mark.parametrize("labels", ["numeric", "text"])
    def test_stats__shared_grouping(self, mixed_dataset: xr.Dataset, labels: str) -> None:
        """
        Checks that Dataset statistics on points share grouping calculation.

        We need this test for finding a grouping field in the Dataset and using it for several selected point variables.
        """

        # We add a numeric or text grouping attribute alongside the point values
        groups = np.arange(9) % 2
        categories: list[int | str] = [0, 1] if labels == "numeric" else ["ground", "canopy"]
        if labels == "text":
            groups = np.array(categories, dtype=object)[groups]
        source = mixed_dataset.assign(zone=("point", groups))
        options = {"by": {"zone": "zone"}, "categories": {"zone": categories}}

        # Then group both point variables by the zone attribute and compute stats
        result = source.pc.stats("mean", variables=["point_z", "point_sigma"], **options)

        # We run the DataArray calculation, passing values from both variables at once
        # (Should be equivalent of Dataset behaviour)
        support = source.point_z.assign_coords(point_sigma=source.point_sigma, zone=source.zone)
        expected = support.pc.stats("mean", values={"point_z": "point_z", "point_sigma": "point_sigma"}, **options)

        # Check exact equality
        pd.testing.assert_frame_equal(result, expected, check_exact=True)

        # Compare with NumPy counts for each numeric/text group
        counts = [np.count_nonzero(groups == category) for category in categories]
        np.testing.assert_array_equal(result.loc[categories, ("point_z", "count")], counts)

    @pytest.mark.parametrize(
        "method,options",
        [
            ("crop", {"bbox": (100, 100, 101, 101)}),
            ("clip", {"mask": box(100, 100, 101, 101)}),
            ("subsample", {"subsample": 4, "mask": np.zeros(9, dtype=bool)}),
        ],
    )
    def test_methods__empty_selection(self, mixed_dataset: xr.Dataset, method: str, options: dict[str, Any]) -> None:
        """Checks that empty point selections preserve variable dtypes and leave independent rasters unchanged."""

        # We add integer classifications to check their dtype after selecting no rows
        source = mixed_dataset.assign(classification=("point", np.arange(9, dtype=np.uint8)))

        # Select with disjoint bounds or an all-false mask and construct an empty reference by row position
        result = getattr(source.pc, method)(**options)
        expected = source.isel(point=slice(0, 0))

        # Compare with the empty reference, including independent rasters, and check the classification dtype
        assert_xarray_equal(result, expected)
        assert result.sizes["point"] == 0
        assert result.classification.dtype == np.uint8

    def test_to_file__selected_variables(self, mixed_dataset: xr.Dataset, tmp_path: Path) -> None:
        """
        Checks that to_file() writes selected point values and shared coordinates without independent rasters.

        We need this test for writing a selected Dataset point field rather than automatically using the first one.
        """

        import_optional("pyarrow")

        # We write only point_sigma to check that the writer uses the selected variable
        filename = tmp_path / "points.parquet"
        mixed_dataset.pc.to_file(filename, variables=["point_sigma"], chunks=4)

        # Reopen all stored columns as a Dataset
        reopened = gu.open_pointcloud(filename, columns="all", as_type="dataset")

        # Check that only the selected point values were written, with their original coordinates and CRS
        assert reopened.pc.variables == ["point_sigma"]
        np.testing.assert_array_equal(reopened.point_sigma.data, mixed_dataset.point_sigma.data)
        assert reopened.point_sigma.pc.georeferenced_coords_equal(mixed_dataset.point_sigma)

    def test_to_las__selected_variables(self, mixed_dataset: xr.Dataset, tmp_path: Path) -> None:
        """
        Checks that selected Dataset fields supply LAS elevations in selection order and omit independent rasters.

        We need this test for choosing LAS Z from the first selected variable and saving the remaining ones as extra
        dimensions.
        """

        laspy = import_optional("laspy")

        # We select uncertainty before elevation to check that the writer uses the requested order for LAS Z
        names = ["point_sigma", "point_z"]
        filename = tmp_path / "selected.las"
        mixed_dataset.pc.to_las(str(filename), variables=names, chunks=4, scales=(0.001, 0.001, 0.001))

        # Read LAS independently and compare elevations, extra values, X/Y and CRS with the selected fields
        stored = laspy.read(filename)
        np.testing.assert_array_equal(stored.z, mixed_dataset.point_sigma.data)
        np.testing.assert_array_equal(stored.point_z, mixed_dataset.point_z.data)
        np.testing.assert_array_equal(stored.x, mixed_dataset.x_point.data)
        np.testing.assert_array_equal(stored.y, mixed_dataset.y_point.data)
        assert list(stored.point_format.extra_dimension_names) == ["point_z"]
        assert stored.header.parse_crs() == mixed_dataset.point_z.pc.crs

    @pytest.mark.parametrize("partitioned", [False, True])
    def test_to_parquet__partitions(self, mixed_dataset: xr.Dataset, tmp_path: Path, partitioned: bool) -> None:
        """
        Checks that writing Dataset points in chunks preserves their values, dtypes, row labels and CRS.

        We need this test for writing categorical/integer point fields and duplicate row labels to GeoParquet.
        """

        import_optional("pyarrow")
        import_optional("dask")

        # 1/ Create a test mixed Xarray Dataset by modifying the mixed_dataset fixture
        # We add categorical values, large integer IDs and duplicate labels to test their saved values and dtypes
        source = mixed_dataset.assign(
            classification=("point", np.array(["ground", "tree", "other"] * 3, dtype=object)),
            identifier=("point", np.arange(9, dtype=np.uint64) + 2**63),
        ).assign_coords(point=np.arange(9) // 2)

        # Split points into chunks of 4/4/1 to test the final partial write
        lazy = source.chunk({"point": 4, "y": 3, "x": 4})
        filename = tmp_path / ("partitions" if partitioned else "points.parquet")

        # 2/ Write chunked points as partitions or row groups, then reopen lazily
        lazy.pc.to_parquet(filename, chunks=4, partitioned=partitioned)

        # Reopen in chunks and check stored point variables, row count and lazy values
        reopened = gu.open_pointcloud(filename, columns="all", chunks=4, as_type="dataset")
        assert reopened.pc.variables == source.pc.variables
        assert reopened.sizes["point"] == 9
        assert lazy.point_z.chunks and reopened.point_z.chunks

        # 3/ Compare values, metadata and loading behaviour
        # Compute stored points and compare their values, dtypes, labels, coordinates and CRS with the source
        actual = reopened.compute()
        for name in lazy.pc.variables:
            np.testing.assert_array_equal(actual[name].data, source[name].data)
            assert actual[name].dtype == source[name].dtype
        np.testing.assert_array_equal(actual.point.data, source.point.data)
        np.testing.assert_array_equal(actual.x_point.data, source.x_point.data)
        np.testing.assert_array_equal(actual.y_point.data, source.y_point.data)
        assert actual.point_z.pc.crs == mixed_dataset.point_z.pc.crs

        # Check that the source point values and independent rasters are still lazy
        assert lazy.point_z.chunks and lazy.dem.chunks

    @pytest.mark.parametrize(
        "method,options",
        [
            ("subsample", {"subsample": 4, "random_state": 3}),
            ("reproject", {"crs": 4326}),
            ("filter", {"method": "mean", "radius": 2}),
        ],
    )
    def test_chunked_methods__row_labels(self, pointcloud_file: Any, method: str, options: dict[str, Any]) -> None:
        """
        Checks that Dataset point operations leave stored row labels lazy without loading them into a Pandas index.

        We need this additional test for row labels read lazily from a file, because the method combinations use labels
        already loaded in memory.
        """

        import geopandas as gpd

        import_optional("dask")
        from dask.callbacks import Callback

        # 1/ Write and open a GeoParquet point file
        # We create seven points with explicit row labels and write GeoParquet row groups of 3/3/1
        frame = gpd.GeoDataFrame(
            {"height": np.arange(7.0), "sigma": np.linspace(1, 2, 7)},
            geometry=gpd.points_from_xy(np.arange(7.0), np.zeros(7)),
            crs=32631,
            index=pd.Index(np.arange(7) * 10, name="point_id"),
        )
        filename, chunks, column = pointcloud_file(frame, 3)
        eager = gu.open_pointcloud(filename, data_name=column, columns="all", as_type="dataset")

        # 2/ Open in chunks and run the point operation without reading the file
        # Record Dask tasks to check that neither opening nor constructing the result reads point values or labels
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(1)):
            lazy = gu.open_pointcloud(filename, data_name=column, columns="all", chunks=chunks, as_type="dataset")
            result = getattr(lazy.pc, method)(**options)

        # Check that values and labels stay lazy and labels are not loaded into an Xarray index
        assert not tasks
        assert lazy.height.chunks and lazy.point.chunks and result.height.chunks and result.point.chunks
        assert "point" not in result.xindexes

        # 3/ Compare eager outputs and loading behaviour
        # Compute and compare with the eager operation, including original labels, then check the source stays lazy
        expected = getattr(eager.pc, method)(**options)
        assert_xarray_equal(result.compute(), expected)
        assert lazy.height.chunks and lazy.point.chunks


class TestDatasetPointCloudAccessorErrors:
    """Test module for errors/warnings for the point cloud Dataset accessor ``pc``."""

    @pytest.mark.parametrize(
        "method,options,message",
        [
            ("reproject", {"crs": 4326, "inplace": True}, "inplace=True"),
            ("subsample", {"subsample": 4, "as_array": True}, "returns a Dataset"),
            ("subsample", {"subsample": 4, "return_indices": True}, "returns a Dataset"),
            ("stats", {"values": ["point_z"]}, "Use 'variables'"),
        ],
    )
    def test_methods__error_dataset_options(
        self, mixed_dataset: xr.Dataset, method: str, options: dict[str, Any], message: str
    ) -> None:
        """Checks an error is raised for invalid options specific to a Dataset accessor."""

        original = mixed_dataset.copy(deep=True)
        with pytest.raises(ValueError, match=message):
            getattr(mixed_dataset.pc, method)(**options)

        # Check unchanged source dataset after error
        xr.testing.assert_identical(mixed_dataset, original)

    @pytest.mark.parametrize(
        "method,options,variables,message",
        [
            ("subsample", {"subsample": 4}, [], "nonempty"),
            ("variogram", {}, ["missing"], "Unknown"),
            ("grid", {}, ["dem"], "incompatible"),
            ("filter", {"method": "mean", "radius": 2}, ["point_z", "point_z"], "distinct"),
        ],
    )
    def test_methods__error_variables(
        self, mixed_dataset: xr.Dataset, method: str, options: dict[str, Any], variables: list[str], message: str
    ) -> None:
        """Checks an error is raised for invalid variable selections."""

        with pytest.raises(ValueError, match=message):
            getattr(mixed_dataset.pc, method)(variables=variables, **options)

    @pytest.mark.parametrize("method,options", [("subsample", {"subsample": 4}), ("reproject", {"crs": 4326})])
    def test_methods__error_untouched_point_values(
        self, mixed_dataset: xr.Dataset, method: str, options: dict[str, Any]
    ) -> None:
        """Checks an error is raised when an unselected point variable would use changed X/Y coordinates or rows."""

        # We change only point_z but not point_sigma (which shares the same coordinates and rows)
        with pytest.raises(ValueError, match="Untouched variables.*point_sigma"):
            getattr(mixed_dataset.pc, method)(variables=["point_z"], **options)

    @pytest.mark.parametrize("method", ["filter", "grid"])
    def test_methods__error_nonnumeric_values(self, mixed_dataset: xr.Dataset, method: str) -> None:
        """Checks an error is raised when filtering or gridding selected text point values."""

        # We add string values
        source = mixed_dataset.assign(classification=("point", np.array(["ground", "tree", "other"] * 3)))
        options = {"method": "mean", "radius": 2} if method == "filter" else {"ref": source.dem}
        with pytest.raises(TypeError, match="numeric value variables"):
            getattr(source.pc, method)(variables=["classification"], **options)

    @pytest.mark.parametrize("change", ["coordinates", "crs", "labels"])
    def test_subsample__error_mask_metadata(self, mixed_dataset: xr.Dataset, change: str) -> None:
        """
        Checks an error is raised for a point mask with different coordinates, CRS or row labels.

        We need this additional test for comparing a mask's X/Y, CRS and row labels with the Dataset before subsampling.
        """

        # We add explicit point labels and create an aligned mask from the fixture's point values
        source = mixed_dataset.assign_coords(point=np.arange(9) * 10)
        mask = source.point_z.copy(data=np.ones(9, dtype=bool))

        # Modify only the mask's spatial coordinates, CRS or row labels to isolate each alignment check
        if change == "coordinates":
            mask = mask.assign_coords(x_point=mask.x_point + 1)
        elif change == "crs":
            mask.attrs["crs"] = "EPSG:4326"
        else:
            mask = mask.assign_coords(point=mask.point.data[::-1])
        message = {"coordinates": "mask coordinates differ", "crs": "same CRS", "labels": "mask labels differ"}

        # Subsample with the mismatched mask and check an error is raised before rows are selected
        with pytest.raises(ValueError, match=message[change]):
            source.pc.subsample(4, mask=mask, random_state=3)

    def test_variogram__error_nonnumeric_values(self, mixed_dataset: xr.Dataset) -> None:
        """
        Checks an error is raised before sampling when a selected variogram variable contains text.

        We need this test for checking all point fields before sampling when numeric and text values share a Dataset.
        """

        # We add text classifications alongside numeric point values
        source = mixed_dataset.assign(classification=("point", np.array(["ground", "tree", "other"] * 3)))

        # Request variograms for all point variables and check an error is raised for the text values
        with pytest.raises(TypeError, match="numeric value variables"):
            source.pc.variogram()

    def test_variogram__error_insufficient_finite_values(self, mixed_dataset: xr.Dataset) -> None:
        """
        Checks an error is raised when a selected variogram variable has fewer than two finite points.

        We need this additional test when the selected field has no finite pairs but another Dataset field has usable
        values.
        """

        # We copy the mixed dataset and replace every uncertainty value with NaN, leaving elevations finite
        source = mixed_dataset.copy(deep=True)
        source.point_sigma.data[:] = np.nan

        # Select only uncertainty values and check an error is raised because no finite pairs can be sampled
        with pytest.raises(ValueError, match="At least two finite points"):
            source.pc.variogram(variables=["point_sigma"], n_pairs=12)

    def test_accessor__error_ambiguous_coordinates(self, mixed_dataset: xr.Dataset) -> None:
        """
        Checks an error is raised when the Dataset contains two independent point dimensions with their own X/Y.

        We need this additional test for requiring one shared point dimension when a Dataset contains several X/Y sets.
        """

        # We add point values on a second dimension with their own X/Y coordinates
        source = mixed_dataset.assign_coords(x_other=("other", [1, 2]), y_other=("other", [3, 4]))
        source["other_z"] = ("other", [5, 6])

        # Request subsampling and check an error is raised because the point dimension is ambiguous
        with pytest.raises(ValueError, match="ambiguous"):
            source.pc.subsample(4)

    @pytest.mark.parametrize(
        "method,options", [("reproject", {"crs": 32632}), ("filter", {"method": "mean", "radius": 2})]
    )
    @pytest.mark.parametrize("name", ["point_sigma", "x_point"])
    def test_methods__error_incompatible_crs(
        self, mixed_dataset: xr.Dataset, method: str, options: dict[str, Any], name: str
    ) -> None:
        """Checks an error is raised for conflicting CRS metadata on shared point values or coordinates."""

        # We copy the mixed dataset and give one point value or X coordinate a conflicting CRS
        source = mixed_dataset.copy(deep=True)
        source[name].attrs["crs"] = "EPSG:4326"

        # Run the shared point operation and check an error is raised for inconsistent CRS metadata
        with pytest.raises(ValueError, match="same CRS"):
            getattr(source.pc, method)(**options)

    def test_grid__error_independent_raster_coordinates(self, mixed_dataset: xr.Dataset) -> None:
        """
        Checks an error is raised when a new point grid would change an independent raster's coordinates.

        We need this test for preventing gridded Dataset points from replacing X/Y coordinates still used by an
        independent raster.
        """

        # Grid at a coarser resolution and check an error is raised because the independent DEM shares those axes
        with pytest.raises(ValueError, match="Untouched variables.*dem"):
            mixed_dataset.pc.grid(res=2, resampling="nearest")

    def test_grid__error_unknown_lazy_extent(self, mixed_dataset: xr.Dataset) -> None:
        """
        Checks an error is raised when gridding lazy points without explicit bounds.

        We need this additional test for requiring grid bounds when finding the point extent would load lazy X/Y.
        """

        import_optional("dask")
        from dask.callbacks import Callback

        # We chunk the points so finding their extent would require computing X/Y coordinates
        lazy = mixed_dataset.chunk({"point": 4})

        # Request a grid without bounds and check an error is raised while recording any Dask tasks
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(1)):
            with pytest.raises(ValueError, match="explicit bounds"):
                lazy.pc.grid(res=2, resampling="nearest")

        # Check that no tasks ran and point values and coordinates stayed lazy
        assert not tasks
        assert lazy.point_z.chunks and lazy.x_point.chunks

    @pytest.mark.parametrize("method", ["crop", "clip"])
    def test_methods__error_lazy_coordinates(self, mixed_dataset: xr.Dataset, method: str) -> None:
        """Checks an error is raised before implicitly computing lazy coordinates for row selection."""

        import_optional("dask")
        from dask.callbacks import Callback

        # We chunk point values and X/Y so finding the selected row count would require reading coordinates
        lazy = mixed_dataset.chunk({"point": 4})

        # Request a spatial selection and check an error is raised while recording any Dask tasks
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(1)):
            with pytest.raises(ValueError, match="compute coordinates"):
                getattr(lazy.pc, method)((0, 0, 3, 4))

        # Check that no tasks ran and point values and coordinates stayed lazy
        assert not tasks
        assert lazy.point_z.chunks and lazy.x_point.chunks

    def test_subsample__error_lazy_mask(self, mixed_dataset: xr.Dataset) -> None:
        """
        Checks an error is raised for subsampling with a lazy mask whose selected row count is unknown.

        We need this test for requiring a loaded mask when Dataset subsampling must know how many point rows it selects.
        """

        import_optional("dask")
        from dask.callbacks import Callback

        # We chunk the points so a mask calculated from their values has an unknown selected row count
        lazy = mixed_dataset.chunk({"point": 4})

        # Subsample with the lazy mask and check an error is raised while recording any Dask tasks
        tasks = []
        with Callback(pretask=lambda *args: tasks.append(1)):
            with pytest.raises(ValueError, match="compute the mask"):
                lazy.pc.subsample(4, mask=lazy.point_z > 2)

        # Check that no tasks ran and the source point values stayed lazy
        assert not tasks
        assert lazy.point_z.chunks
