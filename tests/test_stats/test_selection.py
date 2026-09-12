"""Tests selecting statistic values and masks on a common support."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio as rio
from pyproj import CRS
from rasterio.transform import from_origin
from shapely.geometry import box

import geoutils as gu
from geoutils._misc import import_optional
from geoutils.multiproc import ClusterGenerator, MultiprocConfig


@pytest.fixture(params=["raster", "gpkg", "las"])
def stats_file(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[Any, Any, dict[str, Any], Any]:
    """Write raster or point files with two value columns and one grouping variable to disk."""

    # Create two value columns with nodata values at different locations and one integer grouping column
    positions = np.arange(48)
    first = positions.astype(float) + 1
    second = positions.astype(float) ** 2 + 100
    first[[3, 19]] = np.nan
    second[[8, 27, 43]] = np.nan
    groups = positions % 8
    if request.param == "raster":
        # Write a raster file to disk with the two value columns and grouping column stored as separate bands
        filename = tmp_path / "statistics.tif"
        values = np.stack((first, second, groups)).reshape(3, 6, 8)
        gu.Raster.from_array(values, from_origin(500000, 5100006, 1, 1), 32633, nodata=np.nan).to_file(filename)
        source = gu.Raster(filename)
        reference = gu.Raster(filename, load_data=True)
        return source, reference, {"first": 1, "second": 2}, 3

    filename = tmp_path / f"statistics.{request.param}"
    if request.param == "las":
        # Write a LAS file to disk with values in attributes separate from the point coordinates
        laspy = import_optional("laspy")
        header = laspy.LasHeader(point_format=6, version="1.4")
        header.add_crs(CRS.from_epsg(32633))
        for name, dtype in (("first", "float64"), ("second", "float64"), ("group", "int32")):
            header.add_extra_dim(laspy.ExtraBytesParams(name=name, type=dtype))
        records = laspy.LasData(header)
        records.x, records.y, records.z = 500000 + positions, 5100000 + positions, positions + 1000
        records.first, records.second, records.group = first, second, groups
        records.write(filename)
    else:
        # Write a GeoPackage file to disk with both value columns and the grouping column
        dataframe = gpd.GeoDataFrame(
            {"first": first, "second": second, "group": groups},
            geometry=gpd.points_from_xy(500000 + positions, 5100000 + positions),
            crs=32633,
        )
        dataframe.to_file(filename, index=False)

    # Open one unloaded source for Multiproc and one eager source for the expected result
    source = gu.PointCloud(filename, data_column="first")
    reference = gu.PointCloud(filename, data_column="first")
    reference.load(columns="all")
    return source, reference, {"first": "first", "second": "second"}, "group"


class TestSelection:
    """
    Checks the values and masks accepted by stats().

    - Rasters and point clouds can select bands, columns or plain Xarrays.
    - NumPy and tabular masks use positions on the common support.
    - Dask and Multiproc behavior is covered by TestSelectionChunked below.
    """

    @pytest.mark.parametrize("grouped", [False, True])
    def test_stats__band_selection(self, grouped: bool) -> None:
        """Checks that the values argument selects the requested raster band with and without groups."""

        # Give the two bands different means so selecting the default band would fail
        values = np.arange(1, 7, dtype=float).reshape(2, 3)
        raster = gu.Raster.from_array(
            np.stack((values, values + 100)),
            transform=rio.transform.from_origin(0, 2, 1, 1),
            crs=4326,
        )
        options = {}
        if grouped:
            options = {"by": {"zone": np.array([[0, 0, 0], [1, 1, 1]])}, "categories": {"zone": [0, 1]}}

        # Select the second band through the raster method
        selected = raster.stats("mean", values=2, **options)

        # Check the selected band's whole mean or separate row means
        if grouped:
            np.testing.assert_allclose(selected[("band_2", "mean")], [102, 105])
        else:
            assert selected == pytest.approx(103.5)

    @pytest.mark.parametrize("source_type", ["array", "raster", "pointcloud"])
    @pytest.mark.parametrize("grouped", [False, True])
    def test_stats__mask_selection(self, source_type: str, grouped: bool) -> None:
        """Checks that mask selects the same values and preserves total and valid counts across input types."""

        # Place one nodata value inside the mask to distinguish total and valid inlier counts
        values = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
        keep = np.array([[True, True, False], [True, False, True]])
        groups = np.array([[0, 0, 0], [1, 1, 1]])
        source: Any = values
        if source_type == "raster":
            source = gu.Raster.from_array(values, rio.transform.from_origin(0, 2, 1, 1), 4326, nodata=np.nan)
        elif source_type == "pointcloud":
            source = gu.PointCloud.from_xyz(np.arange(values.size), np.zeros(values.size), values.ravel(), crs=4326)
            keep, groups = keep.ravel(), groups.ravel()

        # Apply the same mask to either a complete summary or the same two groups
        options: dict[str, Any] = {}
        statistics: str | list[str] = "all"
        if grouped:
            options = {"by": {"zone": groups}, "categories": {"zone": [0, 1]}}
            statistics = ["mean", "totalcount"]
        masked = gu.stats.stats(source, statistics, mask=keep, **options)

        # Check counts before and after the mask for the summary and for each group
        if grouped:
            np.testing.assert_allclose(masked.xs("count", axis=1, level="statistic").iloc[:, 0], [1, 2])
            np.testing.assert_allclose(masked.xs("totalcount", axis=1, level="statistic").iloc[:, 0], [2, 2])
            np.testing.assert_allclose(masked.xs("mean", axis=1, level="statistic").iloc[:, 0], [1, 5])
        else:
            assert masked["Mean"] == pytest.approx(11 / 3)
            assert masked["Valid count"] == 5
            assert masked["Total count"] == 6
            assert masked["Valid inlier count"] == 3
            assert masked["Total inlier count"] == 4
            assert masked["Percentage inlier points"] == 60
            assert masked["Percentage valid inlier points"] == 75

    def test_stats__point_column_selection(self) -> None:
        """Checks that direct point cloud summaries use the active or explicitly selected column."""

        # Give the additional column a distinct scale so active-column fallback is detectable
        pointcloud = gu.PointCloud.from_xyz(np.arange(4), np.zeros(4), np.arange(1, 5), crs=4326)
        pointcloud.ds["temperature"] = [10.0, 20.0, 30.0, 40.0]

        # Calculate default and selected summaries through both public entry points
        default = gu.stats.stats(pointcloud)
        selected = gu.stats.stats(pointcloud, "mean", values="temperature")
        method_selected = pointcloud.stats("mean", values="temperature")
        method_masked = pointcloud.stats("mean", values="temperature", mask=np.array([True, True, False, False]))

        # Match the active-column and auxiliary-column means independently
        assert default["Mean"] == pytest.approx(2.5)
        assert selected == method_selected == pytest.approx(25)
        assert method_masked == pytest.approx(15)

    @pytest.mark.parametrize("grouped", [False, True])
    def test_stats__point_inputs_from_plain_xarrays(self, grouped: bool) -> None:
        """Checks that point values, masks and groups in plain Xarrays follow the source's point order."""

        import xarray as xr

        # Use a nonspatial dimension so the point cloud supplies all location information
        pointcloud = gu.PointCloud.from_xyz(np.arange(4), np.zeros(4), np.arange(4), crs=4326)
        values = xr.DataArray([10.0, 20.0, 30.0, 40.0], dims="point")
        keep = xr.DataArray([True, False, True, True], dims="point")
        groups = xr.DataArray([0, 0, 1, 1], dims="point")
        options = {"by": {"zone": groups}, "categories": {"zone": [0, 1]}} if grouped else {}

        # Use plain Xarrays for selected values, the user mask, and the optional grouping variable
        result = pointcloud.stats("mean", values={"temperature": values}, mask=keep, **options)

        # Exclude the second point, leaving one value in the first group and two in the second
        if grouped:
            np.testing.assert_allclose(result["temperature"], [[1, 10], [2, 35]])
        else:
            assert result == pytest.approx((10 + 30 + 40) / 3)

    @pytest.mark.parametrize("grouped", [False, True])
    @pytest.mark.parametrize("source_type", ["array", "raster"])
    def test_stats__grid_inputs_from_plain_xarrays(self, grouped: bool, source_type: str) -> None:
        """Checks that Xarray dimensions named x and y do not require spatial coordinates for statistics."""

        import xarray as xr

        # Give values, groups and the mask spatial dimension names without defining coordinates or a CRS
        values = xr.DataArray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dims=("y", "x"))
        keep = xr.DataArray([[True, False, True], [False, True, True]], dims=("y", "x"))
        groups = xr.DataArray([[0, 0, 0], [1, 1, 1]], dims=("y", "x"))
        options = {"by": {"zone": groups}, "categories": {"zone": [0, 1]}} if grouped else {}

        # Use these arrays directly or select them as values on an existing raster grid
        if source_type == "array":
            result = gu.stats.stats(values, "mean", mask=keep, **options)
            value_name = "value"
        else:
            raster = gu.Raster.from_array(np.zeros((2, 3)), rio.transform.from_origin(0, 2, 1, 1), crs=4326)
            result = raster.stats("mean", values={"temperature": values}, mask=keep, **options)
            value_name = "temperature"

        # The mask leaves two values in each row, giving row means of 20 and 55 and a whole mean of 37.5
        if grouped:
            np.testing.assert_allclose(result[value_name], [[2, 20], [2, 55]])
        else:
            assert result == pytest.approx(37.5)

    @pytest.mark.parametrize("container", ["series", "dataframe", "dask_series"])
    def test_stats__tabular_values_and_masks(self, container: str) -> None:
        """Checks that tabular inputs apply masks by position and return the expected shape and counts."""

        # 1/ Prepare tabular values, masks and groups
        # Use different row labels to check positional selection, with one nodata value inside the mask
        labels = np.arange(6) + 10
        source: Any = pd.Series([1.0, np.nan, 3.0, 4.0, 5.0, 6.0], index=labels)
        keep: Any = pd.Series([True, True, False, True, False, True], index=labels[::-1])
        groups: Any = pd.Series([0, 0, 0, 1, 1, 1], index=labels + 100)
        if container == "dataframe":
            source, keep, groups = (array.to_frame() for array in (source, keep, groups))
        elif container == "dask_series":
            import_optional("dask")
            import dask.array as da
            import dask.dataframe as dd

            source, keep, groups = (
                dd.from_pandas(array, npartitions=2, sort=False) for array in (source, keep, groups)
            )

        # 2/ Calculate masked summaries and groups
        # Request the callable's input shape as well as counts before and after selection
        statistics: list[str | Callable[[Any], Any]] = [
            "mean",
            "validcount",
            "validinliercount",
            "totalinliercount",
            np.shape,
        ]
        summary = gu.stats.stats(source, statistics, mask=keep)
        grouped = gu.stats.stats(
            source,
            ["mean", "totalcount"],
            by={"zone": groups},
            categories={"zone": [0, 1]},
            mask=keep,
        )

        # 3/ Check shape, calculation result and the selected observations
        # Dask Series return a computed summary; DataFrame summaries use their original two-dimensional shape
        if container == "dask_series":
            assert not isinstance(summary["mean"], da.Array)
        assert summary["shape"] == ((6, 1) if container == "dataframe" else (6,))
        assert summary["mean"] == pytest.approx(11 / 3)
        assert summary["validcount"] == 5
        assert summary["validinliercount"] == 3
        assert summary["totalinliercount"] == 4

        # The first group has one finite value in two selected locations; the second has two finite values
        np.testing.assert_allclose(grouped["value"], [[1, 1, 2], [2, 5, 2]])

    @pytest.mark.parametrize("grouped", [False, True])
    def test_stats__error_tabular_inputs(self, grouped: bool) -> None:
        """Checks that tabular statistics reject invalid masks and mixed Dask and Multiproc backends."""

        # A DataFrame is a two-dimensional value array, so each grouper must use that same shape
        values = pd.DataFrame(np.arange(6, dtype=float).reshape(2, 3))
        options: dict[str, Any] = (
            {"by": {"zone": np.zeros(values.shape, dtype=int)}, "categories": {"zone": [0]}} if grouped else {}
        )

        # Reject a row-only mask and a numeric mask rather than broadcasting or treating nonzero values as True
        with pytest.raises(
            ValueError, match="Argument ``mask`` must be boolean and contain one value per input location"
        ):
            gu.stats.stats(values, "mean", mask=pd.Series([True, False]), **options)
        with pytest.raises(
            ValueError, match="Argument ``mask`` must be boolean and contain one value per input location"
        ):
            gu.stats.stats(values, "mean", mask=pd.DataFrame(np.ones(values.shape, dtype=int)), **options)

        # Check that a Dask Series is rejected before the Multiproc backend starts any workers
        import_optional("dask")
        import dask.dataframe as dd

        lazy_values = dd.from_pandas(pd.Series(np.arange(6, dtype=float)), npartitions=2)
        options = {"by": {"zone": np.zeros(6, dtype=int)}, "categories": {"zone": [0]}} if grouped else {}
        with pytest.raises(ValueError, match="Dask inputs cannot be combined with Multiprocessing"):
            gu.stats.stats(lazy_values, "mean", mp_config=MultiprocConfig(chunks=2), **options)


class TestSelectionChunked:
    """
    Checks stats() input selection with Dask and Multiproc inputs.

    Dask and Multiproc results are compared with eager calculations. Raster and point cloud files stay unloaded while
    their values, masks and common support are read in chunks.
    """

    def test_stats__masked_dask_summary_is_computed(self) -> None:
        """Checks that a masked Dask raster gives computed values and the same counts as eager data."""

        # Use mismatched data and mask chunks so masking also checks automatic chunk alignment
        import_optional("dask")
        import dask.array as da
        import xarray as xr

        values = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
        keep = np.array([[True, True, False], [True, False, True]])
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 2, 1, 1), 4326, nodata=np.nan)
        lazy_raster = raster.to_xarray().chunk({"x": 2, "y": 1}).rst
        lazy_keep = xr.DataArray(da.from_array(keep, chunks=(2, 1)), dims=("y", "x"))
        statistics = ["mean", "std", "validcount", "validinliercount", "totalinliercount"]
        assert isinstance(lazy_raster.data, da.Array) and isinstance(lazy_keep.data, da.Array)

        # Check that stats() computes the Dask summary immediately, as it does for grouped results
        result = lazy_raster.stats(statistics, mask=lazy_keep)
        assert not isinstance(result["mean"], da.Array)

        # Compare the result with the same masked eager raster
        expected = raster.stats(statistics, mask=keep)
        assert result == pytest.approx(expected)
        assert isinstance(lazy_raster.data, da.Array) and isinstance(lazy_keep.data, da.Array)

    def test_stats__dask_point_vector_mask(self, tmp_path: Path) -> None:
        """Checks that eager and Dask points agree on points lying along a polygon boundary."""

        # Write a point file to disk with two points inside the polygon, one on its edge, and one outside it
        import geopandas as gpd
        from shapely.geometry import box

        import_optional("dask_geopandas", package_name="dask-geopandas")
        points = gu.PointCloud.from_xyz(
            np.array([0.5, 1.5, 2.0, 2.5]), np.ones(4), np.array([10.0, 20.0, 30.0, 40.0]), crs=32633
        )
        mask = gu.Vector(gpd.GeoDataFrame(geometry=[box(0, 0, 2, 2)], crs=points.crs))
        filename = tmp_path / "masked_points.gpkg"
        points.to_file(filename)
        lazy_points = gu.open_pointcloud(str(filename), data_column=points.data_column, chunks=2).pc
        assert not lazy_points.is_loaded

        # Calculate the same masked summary from Dask partitions and eager points
        statistics = ["mean", "validinliercount"]
        expected = points.stats(statistics, mask=mask)
        result = lazy_points.stats(statistics, mask=mask)

        # Only the first two values contribute; the point on the polygon edge remains excluded
        assert result == pytest.approx(expected)
        assert result == pytest.approx({"mean": 15, "validinliercount": 2})
        assert not lazy_points.is_loaded

    def test_stats__mixed_eager_and_dask_values(self) -> None:
        """Checks that one stats() call returns computed summaries for mixed eager and Dask values."""

        # Select the same locations from eager and Dask values with distinct scales
        import_optional("dask")
        import dask.array as da

        eager = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0])
        lazy = da.from_array(eager * 10, chunks=2)
        keep = np.array([True, True, False, True, False, True])

        # Calculate both value arrays in the same Dask call
        result = gu.stats.stats({"eager": eager, "lazy": lazy}, ["mean", "validinliercount"], mask=keep)
        assert isinstance(result["eager"]["mean"], float)
        assert isinstance(result["lazy"]["mean"], float)

        # Both values select three finite locations, with means differing by the known factor of ten
        assert result["eager"] == pytest.approx({"mean": 11 / 3, "validinliercount": 3})
        assert result["lazy"] == pytest.approx({"mean": 110 / 3, "validinliercount": 3})
        assert isinstance(lazy, da.Array)

    @pytest.mark.parametrize("workers", [False, True])
    @pytest.mark.parametrize("grouped", [False, True])
    @pytest.mark.parametrize("exact", [False, True])
    def test_stats__file_values_match_eager(
        self,
        stats_file: tuple[Any, Any, dict[str, Any], Any],
        workers: bool,
        grouped: bool,
        exact: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Checks that Multiproc statistics match eager while the complete source stays on disk."""

        # Select both value columns so their separate nodata locations produce different counts and statistics
        source, reference, values, group = stats_file
        # Include statistics that require every finite value from a group at once
        statistics = ["median", np.nanmean] if exact else ["mean", "std", "min", "max", "sumofsquares"]
        options = {"by": {"range": group}, "bins": {"range": [-0.5, 2.5, 5.5, 7.5]}} if grouped else {}
        expected = reference.stats(statistics, values=values, **options)

        # Prevent full-source reads so file statistics must use raster windows or point row ranges
        def reject_load(*args: Any, **kwargs: Any) -> None:
            """Reject a full read while file statistics should be reading only the requested blocks."""

            raise AssertionError("Statistics must read file blocks without calling source.load().")

        with monkeypatch.context() as guarded:
            guarded.setattr(type(source), "load", reject_load)
            with ClusterGenerator("multi" if workers else "basic", nb_workers=2) as cluster:
                config = MultiprocConfig(chunks=(2, 3), cluster=cluster)
                result = source.stats(statistics, values=values, mp_config=config, **options)

        # Compare the complete grouped table or every global result with the eager stats() call
        if grouped:
            pd.testing.assert_frame_equal(result, expected, check_dtype=False)
        else:
            assert result.keys() == expected.keys()
            for name in expected:
                assert result[name] == pytest.approx(expected[name])
        assert not source.is_loaded

    @pytest.mark.parametrize("workers", [False, True])
    def test_stats__automatic_bins_sampling_and_masks(
        self, stats_file: tuple[Any, Any, dict[str, Any], Any], workers: bool
    ) -> None:
        """Checks that automatic bins and sampling return full group masks while the source is not loaded."""

        # Exclude the highest group coordinate so automatic edges must be inferred after applying the mask
        source, reference, values, group = stats_file
        keep = np.arange(48) % 8 != 7
        if isinstance(source, gu.Raster):
            keep = keep.reshape(source.shape)
        options = {
            "values": values,
            "by": {"range": group},
            "bins": {"range": 3},
            "mask": keep,
            "subsample": 12,
            "random_state": 42,
            "return_masks": True,
        }
        expected, expected_masks = reference.stats(["mean", "std"], **options)

        # Find automatic bin edges, select a repeatable sample, and calculate its statistics in file blocks
        with ClusterGenerator("multi" if workers else "basic", nb_workers=2) as cluster:
            config = MultiprocConfig(chunks=(2, 3), cluster=cluster)
            result, masks = source.stats(["mean", "std"], mp_config=config, **options)
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)
        assert not source.is_loaded

        # Requested masks contain every group location, including locations outside the twelve-point sample
        group_location_count = 0
        for key in result.index:
            mask = masks[key]
            assert np.array_equal(mask.data, expected_masks[key].data)
            group_location_count += int(np.count_nonzero(mask.data))
            assert not source.is_loaded
        assert group_location_count == int(keep.sum()) == 42

    @pytest.mark.parametrize("workers", [False, True])
    def test_stats__file_mask_summary_counts(self, workers: bool, tmp_path: Path) -> None:
        """Checks that unloaded boolean masks preserve original valid counts and selected inlier counts."""

        # Write a value raster and boolean mask raster to disk, with one value nodata cell inside and outside the mask
        values = np.arange(1, 25, dtype=float).reshape(4, 6)
        values[0, 0], values[3, 5] = np.nan, np.nan
        keep = np.ones(values.shape, dtype=bool)
        keep[2:, 4:] = False
        transform = from_origin(0, 4, 1, 1)
        value_filename, mask_filename = tmp_path / "values.tif", tmp_path / "mask.tif"
        gu.Raster.from_array(values, transform, 32633, nodata=np.nan).to_file(value_filename)
        gu.Raster.from_array(keep, transform, 32633).to_file(mask_filename)
        source, mask_source = gu.Raster(value_filename), gu.Raster(mask_filename, is_mask=True)
        statistics = ["mean", "validcount", "totalcount", "validinliercount", "totalinliercount"]
        expected = gu.stats.stats(values, statistics, mask=keep)

        # Count the original finite values separately from values retained by the file mask
        with ClusterGenerator("multi" if workers else "basic", nb_workers=2) as cluster:
            config = MultiprocConfig(chunks=(3, 5), cluster=cluster)
            result = source.stats(statistics, mask=mask_source, mp_config=config)

        # The edge windows have one row or column, and all twenty selected cells include one nodata cell
        assert result == pytest.approx(expected)
        assert result["validcount"] == 22 and result["totalcount"] == 24
        assert result["validinliercount"] == 19 and result["totalinliercount"] == 20
        assert not source.is_loaded and not mask_source.is_loaded

    @pytest.mark.parametrize("nullable_integer", [False, True])
    def test_stats__sampled_point_missing_values(self, nullable_integer: bool, tmp_path: Path) -> None:
        """Checks that sampled point summaries read floating and nullable integer columns containing nodata."""

        # Write a point file to disk with a nodata value in the final block and an optional nullable integer column
        values = (
            pd.array([1, 2, 3, 4, 5, None], dtype="Int64") if nullable_integer else np.array([1, 2, 3, 4, 5, np.nan])
        )
        dataframe = gpd.GeoDataFrame(
            {"value": values}, geometry=gpd.points_from_xy(np.arange(6), np.arange(6)), crs=32633
        )
        filename = tmp_path / "missing_points.gpkg"
        dataframe.to_file(filename, index=False)
        source = gu.PointCloud(filename, data_column="value")

        # Selecting all six rows includes the row with nodata in the sample without changing the mean
        statistics = ["mean", "min", "max", "sum", "validcount", "totalcount"]
        config = MultiprocConfig(chunks=2)
        result = source.stats(statistics, subsample=6, random_state=0, mp_config=config)

        # The five finite values sum to fifteen; the file's nodata row contributes only to the total count
        expected = {"mean": 3, "min": 1, "max": 5, "sum": 15, "validcount": 5, "totalcount": 6}
        assert result == pytest.approx(expected)
        assert not source.is_loaded

    @pytest.mark.parametrize("subsample", [1, 4])
    def test_stats__point_infinity_summary(self, subsample: int, tmp_path: Path) -> None:
        """Checks that point summaries include infinity in estimates but count only finite values as valid."""

        # Write a point file to disk with one unmasked infinite value and three finite values
        dataframe = gpd.GeoDataFrame(
            {"value": [1, np.inf, 3, 4]}, geometry=gpd.points_from_xy(np.arange(4), np.arange(4)), crs=32633
        )
        filename = tmp_path / "infinite_points.gpkg"
        dataframe.to_file(filename, index=False)
        source = gu.PointCloud(filename, data_column="value")

        # A sample size of four reads every row through sampling, while one uses the complete summary directly
        statistics = ["mean", "min", "max", "sum", "validcount", "totalcount"]
        result = source.stats(statistics, subsample=subsample, random_state=0, mp_config=MultiprocConfig(chunks=2))

        # NumPy's NaN-aware estimators include positive infinity, and only three observations are finite
        expected = {"mean": np.inf, "min": 1, "max": np.inf, "sum": np.inf, "validcount": 3, "totalcount": 4}
        assert result == expected
        assert not source.is_loaded

    @pytest.mark.parametrize("kind", ["raster", "points"])
    @pytest.mark.parametrize("mask_mode", ["inside", "outside"])
    def test_stats__vector_masks_and_values(self, kind: str, mask_mode: str, tmp_path: Path) -> None:
        """Checks that file statistics apply vector masks and attributes while preserving all summary counts."""

        # Create twelve values on both sides of a polygon, with one nonfinite value on each side
        values = np.arange(12, dtype=float).reshape(3, 4)
        values[0, 0], values[2, 3] = np.nan, np.nan
        zones = gpd.GeoDataFrame({"rating": [10.0]}, geometry=[box(0, 0, 2, 3)], crs=32631)
        source: Any
        reference: Any
        if kind == "raster":
            # Write a raster file to disk with nodata cells inside and outside the polygon
            path = tmp_path / "masked.tif"
            gu.Raster.from_array(values, from_origin(0, 3, 1, 1), 32631, nodata=np.nan).to_file(path)
            source = gu.Raster(path)
            reference = gu.Raster(path, load_data=True)
        else:
            # Write a point file to disk with nodata values inside and outside the polygon
            x, y = np.meshgrid(np.arange(4) + 0.5, np.arange(3) + 0.5)
            dataframe = gpd.GeoDataFrame(
                {"height": values.ravel()}, geometry=gpd.points_from_xy(x.ravel(), y.ravel()), crs=32631
            )
            path = tmp_path / "masked.gpkg"
            dataframe.to_file(path, index=False)
            source = gu.PointCloud(path, data_column="height")
            reference = gu.PointCloud(dataframe, data_column="height")

        # Use edge tiles with one row or column and retain counts from before the vector mask
        statistics = ["mean", "validcount", "totalcount", "validinliercount", "totalinliercount"]
        expected = reference.stats(statistics, mask=zones, mask_mode=mask_mode)
        with ClusterGenerator("multi", nb_workers=2) as cluster:
            config = MultiprocConfig(chunks=(2, 3), cluster=cluster)
            result = source.stats(statistics, mask=zones, mask_mode=mask_mode, mp_config=config)
            attribute = source.stats("mean", values={"rating": (zones, "rating")}, mp_config=config)

        # Both selections contain five finite values, while the vector attribute is constant where defined
        assert result == pytest.approx(expected)
        assert result["validcount"] == 10 and result["validinliercount"] == 5
        assert result["totalinliercount"] == 6 and attribute == 10
        assert not source.is_loaded

    def test_stats__raster_values_at_file_points(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that Multiproc interpolates raster values at file point locations while both sources stay unloaded."""

        # Create a raster and points on distant, exactly known raster cells
        values = np.arange(48, dtype=float).reshape(6, 8)
        rows, columns = np.array([1, 4, 2, 3, 1, 4]), np.array([1, 6, 4, 2, 5, 3])
        raster_filename, point_filename = tmp_path / "values.tif", tmp_path / "locations.gpkg"
        raster = gu.Raster.from_array(values, from_origin(500000, 5100006, 1, 1), 32633)
        x, y = raster.ij2xy(rows, columns)
        points = gpd.GeoDataFrame(
            {"height": np.zeros(len(rows))},
            geometry=gpd.points_from_xy(x, y),
            crs=32633,
        )

        # Write the raster and point files to disk with values at the matching locations
        raster.to_file(raster_filename)
        points.to_file(point_filename, index=False)
        source, support = gu.Raster(raster_filename), gu.PointCloud(point_filename, data_column="height")
        expected = raster.stats("mean", at=gu.PointCloud(points, data_column="height"), interpolation="nearest")

        # Reject full-file loading so Multiproc must read raster windows and matching point rows
        def reject_load(*args: Any, **kwargs: Any) -> None:
            """Reject full loading when only selected point locations are needed."""

            raise AssertionError("Spatial statistics must read bounded file blocks.")

        with monkeypatch.context() as guarded:
            guarded.setattr(gu.Raster, "load", reject_load)
            guarded.setattr(gu.PointCloud, "load", reject_load)
            with ClusterGenerator("multi", nb_workers=2) as cluster:
                config = MultiprocConfig(chunks=(2, 3), cluster=cluster)
                result = source.stats("mean", at=support, interpolation="nearest", mp_config=config)

        # Compare the Multiproc mean with eager stats() and the exact raster values at the point locations
        assert result == pytest.approx(expected)
        assert result == pytest.approx(values[rows, columns].mean())
        assert not source.is_loaded and not support.is_loaded

    @pytest.mark.parametrize("reordered", [False, True])
    def test_stats__point_files_on_common_support(self, reordered: bool, tmp_path: Path) -> None:
        """Checks that distinct point files must have the same ordered coordinates before their values are reduced."""

        # Create two point datasets with matching coordinates and optionally reorder the common support
        positions = np.arange(8)
        values = gpd.GeoDataFrame(
            {"height": positions + 10.0},
            geometry=gpd.points_from_xy(500000 + positions, 5100000 + positions),
            crs=32633,
        )
        locations = values.copy()
        locations["height"] = 0.0
        if reordered:
            locations = locations.iloc[[0, 1, 3, 2, 4, 5, 6, 7]]

        # Write both point datasets to disk with unrelated values in their data columns
        value_filename, point_filename = tmp_path / "values.gpkg", tmp_path / "locations.gpkg"
        values.to_file(value_filename, index=False)
        locations.to_file(point_filename, index=False)
        source = gu.PointCloud(value_filename, data_column="height")
        support = gu.PointCloud(point_filename, data_column="height")

        # Compare all ordered coordinates in Multiproc row blocks without loading either complete file
        with ClusterGenerator("multi", nb_workers=2) as cluster:
            config = MultiprocConfig(chunks=3, cluster=cluster)
            if reordered:
                with pytest.raises(ValueError, match="ordered support coordinates"):
                    source.stats("mean", at=support, mp_config=config)
            else:
                result = source.stats("mean", at=support, mp_config=config)
                assert result == pytest.approx(values.height.mean())
        assert not source.is_loaded and not support.is_loaded

    def test_stats__file_raster_alignment(self, tmp_path: Path) -> None:
        """Checks that Multiproc aligns an unloaded raster to another raster before calculating statistics."""

        # Write two raster files to disk with a non-default value band and a coarser common support grid
        first = np.arange(96, dtype=float).reshape(8, 12)
        values = np.stack((first, first * 3 + 100))
        source_filename, reference_filename = tmp_path / "values.tif", tmp_path / "reference.tif"
        raster = gu.Raster.from_array(values, from_origin(500000, 5100008, 1, 1), 32633, nodata=-9999)
        reference = gu.Raster.from_array(np.zeros((4, 6)), from_origin(500000, 5100008, 2, 2), 32633)
        raster.to_file(source_filename)
        reference.to_file(reference_filename)
        source, support = gu.Raster(source_filename), gu.Raster(reference_filename)
        expected = raster.stats(["mean", "std"], values=2, at=reference, align="reproject")

        # Reproject to a temporary file and read its selected band in separate reduction workers
        with ClusterGenerator("multi", nb_workers=2) as cluster:
            config = MultiprocConfig(chunks=(3, 4), cluster=cluster)
            result = source.stats(["mean", "std"], values=2, at=support, align="reproject", mp_config=config)

        # The completed statistics match eager alignment, and original sources remain available on disk
        assert result == pytest.approx(expected)
        assert not source.is_loaded and not support.is_loaded
        assert source_filename.exists() and reference_filename.exists()
