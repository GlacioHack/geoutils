"""Tests for spatial error predictors and magnitude maps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin

import geoutils as gu
from geoutils.multiproc import MultiprocConfig
from geoutils.multiproc.cluster import MpCluster
from geoutils.raster.xr_accessor import DataArrayRasterAccessor


class TestPredictors:
    """
    Test module for predictor manipulation on raster and point support.

    Tests on chunked execution with Dask/MP are further below in TestPredictorsChunked, this is only in-memory.
    """

    def test_predict_magnitude__raster_like(self) -> None:
        """Checks predicted magnitude with a Raster passed to ``like``."""

        # Create synthetic raster and error structure
        values = np.ma.array([[1.0, 2.0], [3.0, 4.0]], mask=[[False, True], [False, False]])
        raster = gu.Raster.from_array(values, (1, 0, 0, 0, -1, 2), 32631, nodata=-9999)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])

        # Check predicted errors  match shape + nodata of raster input
        result = structure.predict_magnitude(like=raster)
        assert isinstance(result, gu.Raster)
        np.testing.assert_array_equal(result.to_nanarray(), [[2.0, np.nan], [2.0, 2.0]])
        np.testing.assert_array_equal(raster.to_nanarray(), [[1.0, np.nan], [3.0, 4.0]])

    def test_predict_magnitude__xarray_like(self) -> None:
        """Checks predicted magnitude with a Xarray object passed to ``like``."""

        # Create synthetic dataarray and error structure
        values = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
        raster = DataArrayRasterAccessor.from_array(values, transform=from_origin(10, 20, 2, 2), crs=32631)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])

        # Check predicted errors  match shape + NaNs of raster input
        result = structure.predict_magnitude(like=raster)
        assert result.shape == raster.shape
        assert result.rst.transform == raster.rst.transform
        assert result.rst.crs == raster.rst.crs
        np.testing.assert_array_equal(result.data, [[2.0, np.nan, 2.0], [2.0, 2.0, 2.0]])
        np.testing.assert_array_equal(raster.data, values)

    def test_predict_magnitude__interpolation(self) -> None:
        """Checks that predictor interpolation with variable magnitude."""

        # We create a synthetic raster and variable error magnitude
        values = np.ma.masked_array(np.ones((2, 3)), mask=[[False, True, False], [False, False, False]])
        raster = gu.Raster.from_array(values, transform=from_origin(0, 2, 1, 1), crs=32631)
        quality = np.array([[0.0, 1.0, 2.0], [2.0, 1.0, np.nan]])
        statistics = pd.DataFrame({"std": [1.0, 3.0], "count": [10, 10]}, index=pd.Index([0.0, 2.0], name="quality"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", magnitude)])

        # Check predicted values, we should have NaNs matching that of predictor + raster,
        # and linear interpolation of the STD (yielding 2 for the 5th value)
        result = structure.predict_magnitude({"quality": quality}, like=raster)
        expected = np.array([[1.0, np.nan, 3.0], [3.0, 2.0, np.nan]])
        np.testing.assert_array_equal(result.to_nanarray(), expected)
        np.testing.assert_array_equal(raster.to_nanarray(), [[1.0, np.nan, 1.0], [1.0, 1.0, 1.0]])

    def test_predict_magnitude__point_match_array(self) -> None:
        """Checks that point/array give the same magnitudes."""

        # We create a synthetic point cloud and error structure
        points = gu.PointCloud.from_xyz([0, 1, 2], [5, 6, 7], [10, 10, 10], crs=32631)
        quality = np.array([0.0, 0.5, 1.0])
        points.gdf["quality"] = quality
        statistics = pd.DataFrame({"std": [1.0, 3.0], "count": [10, 10]}, index=pd.Index([0.0, 1.0], name="quality"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", magnitude)])

        # Check both predictions are equal
        from_column = structure.predict_magnitude({"quality": "quality"}, like=points)
        from_array = structure.predict_magnitude({"quality": quality}, like=points)
        np.testing.assert_array_equal(from_column.data, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(from_array.data, from_column.data)
        assert from_column.geometry.equals(points.geometry)
        np.testing.assert_array_equal(points.data, [10, 10, 10])

    def test_predict_magnitude__selected_component(self) -> None:
        """Checks that selecting one error structure component excludes the others."""

        # We create a synthetic error structure and raster
        raster = gu.Raster.from_array(np.ones((1, 3)), transform=from_origin(0, 1, 1, 1), crs=32631)
        quality = np.array([[0.0, 0.5, 1.0]])
        statistics = pd.DataFrame({"std": [1.0, 3.0], "count": [10, 10]}, index=pd.Index([0.0, 1.0], name="quality"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        structure = gu.ErrorStructure([gu.ErrorComponent("terrain", magnitude), gu.ErrorComponent("measurement", 4)])

        # Check the selected component uses its own magnitudes, while the total combines squared errors
        selected = structure.predict_magnitude({"quality": quality}, like=raster, component="terrain")
        combined = structure.predict_magnitude({"quality": quality}, like=raster)
        np.testing.assert_array_equal(selected.to_nanarray(), [[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(combined.to_nanarray(), np.sqrt([[17.0, 20.0, 25.0]]))


class TestPredictorsChunked:
    """Test module for chunked (Dask/MP) error magnitude prediction on raster and point."""

    def test_predict_magnitude__raster_dask_mp_equal(self, tmp_path: Path) -> None:
        """Checks that Dask and MP raster tiles predict the eager magnitude map without loading source files."""

        pytest.importorskip("dask")

        # 1/ We create a raster and error structure to evaluate in-memory for later comparison
        values = np.ma.masked_array(np.ones((5, 7)), mask=np.eye(5, 7, dtype=bool))
        source = gu.Raster.from_array(values, transform=from_origin(0, 5, 1, 1), crs=32631, nodata=-9999)
        quality = gu.Raster.from_array(
            np.tile(np.linspace(0, 1, 7), (5, 1)), transform=source.transform, crs=source.crs
        )
        statistics = pd.DataFrame({"std": [1.0, 2.0], "count": [10, 10]}, index=pd.Index([0.0, 1.0], name="quality"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", magnitude)])

        # We predict in-memory
        expected = structure.predict_magnitude({"quality": quality}, like=source)

        # 2/ Same file creation/opening with MP/Dask
        source_path = tmp_path / "source.tif"
        quality_path = tmp_path / "quality.tif"
        source.to_file(source_path)
        quality.to_file(quality_path)
        file_source = gu.Raster(source_path)
        file_quality = gu.Raster(quality_path)
        dask_source = gu.open_raster(str(source_path), chunks={"y": 3, "x": 3})
        dask_quality = gu.open_raster(str(quality_path), chunks={"y": 3, "x": 3})

        # Prediction with Dask/MP
        lazy = structure.predict_magnitude({"quality": dask_quality}, like=dask_source)

        # Should be unloaded/lazy
        assert not file_source.is_loaded and not file_quality.is_loaded
        assert hasattr(dask_source.data, "compute") and hasattr(dask_quality.data, "compute")
        assert hasattr(lazy.data, "compute")
        assert lazy.data.chunks == ((3, 2), (3, 3, 1))
        with MpCluster({"nb_workers": 2}) as cluster:
            config = MultiprocConfig(chunks=(3, 3), outfile=str(tmp_path / "magnitude.tif"), cluster=cluster)
            multiproc = structure.predict_magnitude({"quality": file_quality}, like=file_source, mp_config=config)
        assert not file_source.is_loaded and not file_quality.is_loaded and not multiproc.is_loaded

        # 3/ Check exact equality of output
        np.testing.assert_array_equal(np.asarray(lazy.compute()), expected.to_nanarray())
        np.testing.assert_array_equal(multiproc.to_nanarray(), expected.to_nanarray())

    @pytest.mark.parametrize("as_type", ["dataarray", "geodataframe"])
    def test_predict_magnitude__point_dask_mp_equal(
        self, as_type: Literal["dataarray", "geodataframe"], tmp_path: Path
    ) -> None:
        """Checks that Dask and MP point partitions predict the eager values in the same row order."""

        pytest.importorskip("dask_geopandas")

        # 1/ We create a point and error structure to evaluate in-memory for later comparison
        points = gu.PointCloud.from_xyz(np.arange(7), np.arange(7), np.ones(7), crs=32631)
        points.gdf["quality"] = np.linspace(0, 1, 7)
        statistics = pd.DataFrame({"std": [1.0, 2.0], "count": [10, 10]}, index=pd.Index([0.0, 1.0], name="quality"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", magnitude)])

        # Prediction
        expected = structure.predict_magnitude({"quality": "quality"}, like=points)

        # 2/ Same file creation/opening with MP/Dask
        source_path = tmp_path / "points.gpkg"
        points.to_file(source_path)
        dask_points = gu.open_pointcloud(str(source_path), data_name=points.data_name, chunks=3, as_type=as_type)
        file_points = gu.PointCloud(source_path, data_name=points.data_name)
        lazy = structure.predict_magnitude({"quality": "quality"}, like=dask_points)
        assert not dask_points.pc.is_loaded and not lazy.pc.is_loaded
        assert hasattr(lazy, "compute")
        with MpCluster({"nb_workers": 2}) as cluster:
            config = MultiprocConfig(chunks=3, outfile=str(tmp_path / "magnitude.gpkg"), cluster=cluster)
            multiproc = structure.predict_magnitude({"quality": "quality"}, like=file_points, mp_config=config)
        assert not file_points.is_loaded and not multiproc.is_loaded

        # 3/ Exact equality across backends
        computed = lazy.compute()
        values = computed.data if as_type == "dataarray" else computed[points.data_name]
        np.testing.assert_array_equal(values, expected.data)
        np.testing.assert_array_equal(multiproc.data, expected.data)

    def test_predict_magnitude__dask_no_predictors(self) -> None:
        """Checks that a constant magnitude leaves a Dask raster lazy."""

        dask = pytest.importorskip("dask.array")

        # Synthetic raster with one NaN
        values = np.ones((3, 4))
        values[1, 2] = np.nan
        transform = from_origin(0, 3, 1, 1)
        eager = DataArrayRasterAccessor.from_array(values, transform=transform, crs=32631)
        lazy = DataArrayRasterAccessor.from_array(
            dask.from_array(values, chunks=(2, 3)), transform=transform, crs=32631
        )
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])

        # Predict in-memory and Dask
        expected = structure.predict_magnitude(like=eager)
        result = structure.predict_magnitude(like=lazy)
        assert hasattr(lazy.data, "compute") and hasattr(result.data, "compute")
        assert result.data.chunks == ((2, 1), (3, 1))

        # Both should match exactly
        np.testing.assert_array_equal(np.asarray(result.compute()), np.asarray(expected))

    def test_predict_magnitude__multiband_raster_chunks(self) -> None:
        """Checks that each band has its own predictor values and missing pixel mask across Dask tiles."""

        pytest.importorskip("dask")

        # We create a 2-band raster with some NaNs
        mask = np.zeros((2, 3, 4), dtype=bool)
        mask[0, 0, 0] = True
        mask[1, 2, 3] = True
        source = gu.Raster.from_array(
            np.ma.masked_array(np.ones((2, 3, 4)), mask=mask), transform=from_origin(0, 3, 1, 1), crs=32631
        )
        quality = np.stack((np.zeros((3, 4)), np.ones((3, 4))))
        statistics = pd.DataFrame({"std": [1.0, 2.0], "count": [10, 10]}, index=pd.Index([0.0, 1.0], name="quality"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", magnitude)])

        # Run magnitude prediction
        expected = structure.predict_magnitude({"quality": quality}, like=source)
        lazy = structure.predict_magnitude({"quality": quality}, like=source, chunksizes=(2, 3))
        assert expected.shape == source.shape == lazy.rst.shape
        assert expected.data.shape == source.data.shape == lazy.data.shape
        assert hasattr(lazy.data, "compute")

        # We check each band has its own NaNs and magnitude
        reference = np.where(mask, np.nan, np.where(quality == 0, 1.0, 2.0))
        np.testing.assert_array_equal(expected.to_nanarray(), reference)
        np.testing.assert_array_equal(np.asarray(lazy.compute()), reference)

    def test_predict_magnitude__dask_predictors_select_lazy_output(self) -> None:
        """Checks that Dask predictors produce lazy raster and point maps without explicit chunk sizes."""

        dask = pytest.importorskip("dask.array")
        pytest.importorskip("dask_geopandas")

        # The same quality values predict an error from one to two on each spatial input
        quality = np.linspace(0, 1, 7)
        statistics = pd.DataFrame({"std": [1.0, 2.0], "count": [10, 10]}, index=pd.Index([0.0, 1.0], name="quality"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", magnitude)])
        raster = gu.Raster.from_array(np.ones((2, 7)), transform=from_origin(0, 2, 1, 1), crs=32631)
        points = gu.PointCloud.from_xyz(np.arange(7), np.arange(7), np.ones(7), crs=32631)

        # Dask predictor chunks select lazy output even with eager source data
        raster_predictor = dask.from_array(np.tile(quality, (2, 1)), chunks=(1, 3))
        point_predictor = dask.from_array(quality, chunks=3)
        raster_result = structure.predict_magnitude({"quality": raster_predictor}, like=raster)
        point_result = structure.predict_magnitude({"quality": point_predictor}, like=points)
        assert hasattr(raster_result.data, "compute") and not point_result.pc.is_loaded
        assert raster_result.data.chunks == ((1, 1), (3, 3, 1))

        # A scalar predictor also applies to every raster block
        scalar_result = structure.predict_magnitude({"quality": 0.5}, like=raster, chunksizes=(1, 3))
        assert hasattr(scalar_result.data, "compute")
        np.testing.assert_array_equal(np.asarray(scalar_result.compute()), np.full((2, 7), 1.5))

        # Computing both maps matches the same predictions on eager inputs
        eager_raster = structure.predict_magnitude({"quality": np.tile(quality, (2, 1))}, like=raster)
        eager_points = structure.predict_magnitude({"quality": quality}, like=points)
        np.testing.assert_array_equal(np.asarray(raster_result.compute()), eager_raster.to_nanarray())
        np.testing.assert_array_equal(point_result.compute()[points.data_name].to_numpy(), eager_points.data)

    def test_predict_magnitude__point_geometry_z_chunks(self) -> None:
        """Checks that point magnitudes replace geometry Z when no data column is selected."""

        pytest.importorskip("dask_geopandas")

        # We use 3D points
        x = np.arange(7, dtype=float)
        y = np.array([0, 1, 0, 2, 1, 3, 2], dtype=float)
        frame = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y, np.arange(7)), crs=32631)
        points = gu.PointCloud(frame, data_name=None)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 2)])

        # We check both in-memory and lazy write to Z
        expected = structure.predict_magnitude(like=points)
        lazy = structure.predict_magnitude(like=points, chunksizes=3)
        assert not lazy.pc.is_loaded
        result = lazy.compute()
        np.testing.assert_array_equal(result.geometry.x, expected.geometry.x)
        np.testing.assert_array_equal(result.geometry.y, expected.geometry.y)
        np.testing.assert_array_equal(result.geometry.z, expected.geometry.z)

    def test_predict_magnitude__point_array_dask_mp_equal(self, tmp_path: Path) -> None:
        """Checks that Dask and MP align array predictors with uneven point row partitions."""

        pytest.importorskip("dask_geopandas")

        # Synthetic inputs
        points = gu.PointCloud.from_xyz(np.arange(7), np.arange(7), np.ones(7), crs=32631)
        quality = np.linspace(0.0, 1.0, 7)
        statistics = pd.DataFrame({"std": [1.0, 2.0], "count": [10, 10]}, index=pd.Index([0.0, 1.0], name="quality"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", magnitude)])

        # Compute lazy/loaded
        expected = structure.predict_magnitude({"quality": quality}, like=points)
        lazy = structure.predict_magnitude({"quality": quality}, like=points, chunksizes=3)
        assert points.is_loaded and not lazy.pc.is_loaded

        # Worker output stays file-backed until its values are read
        with MpCluster({"nb_workers": 2}) as cluster:
            config = MultiprocConfig(chunks=3, outfile=str(tmp_path / "magnitude.gpkg"), cluster=cluster)
            multiproc = structure.predict_magnitude({"quality": quality}, like=points, mp_config=config)
        assert points.is_loaded and not multiproc.is_loaded

        # All outputs should match exactly
        np.testing.assert_array_equal(lazy.compute()[points.data_name], expected.data)
        np.testing.assert_array_equal(multiproc.data, expected.data)


class TestPredictorsErrors:
    """Test module for errors/warning on manipulating predictors."""

    @pytest.mark.parametrize(
        "groups, error, message",
        [
            ([0.0, 0.0], ValueError, "unique group coordinates"),
            (["low", "high"], TypeError, "continuous numeric groups"),
        ],
    )
    def test_variable_from_grouped_stats__error_invalid_groups(
        self, groups: list[Any], error: type[Exception], message: str
    ) -> None:
        """Checks an error is raised for invalid predictor groups."""

        # Add erroneous group to the predictors
        statistics = pd.DataFrame({"std": [1.0, 2.0], "count": [10, 10]}, index=pd.Index(groups, name="quality"))
        with pytest.raises(error, match=message):
            gu.ErrorMagnitude.variable_from_grouped_stats(statistics)

    def test_variable_from_grouped_stats__min_count(self) -> None:
        """Checks an error is raised when every predictor group has too few observations."""

        # Define two groups below min_count
        statistics = pd.DataFrame({"std": [1.0, 2.0], "count": [1, 2]}, index=pd.Index([0.0, 1.0], name="quality"))
        with pytest.raises(ValueError, match="No finite .* remains after applying min_count=3"):
            gu.ErrorMagnitude.variable_from_grouped_stats(statistics, min_count=3)

    def test_predict_magnitude__error_invalid_support(self) -> None:
        """Checks an error is raised when output support is not a raster/pointcloud."""

        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        with pytest.raises(TypeError, match="like must be a raster or point cloud"):
            structure.predict_magnitude(like=np.ones((2, 2)))

    def test_predict_magnitude__error_missing_chunked_support(self) -> None:
        """Checks an error is raised when like support is not passed for chunked execution."""

        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        with pytest.raises(ValueError, match="like is required for a spatial error magnitude map"):
            structure.predict_magnitude(chunksizes=(1, 1))

    def test_predict_magnitude__error_unknown_component(self) -> None:
        """Checks an error is raised when selected component does not exist in the ErrorStructure."""

        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        raster = gu.Raster.from_array(np.ones((2, 2)), transform=from_origin(0, 2, 1, 1), crs=32631)
        with pytest.raises(KeyError, match="unknown"):
            structure.predict_magnitude(like=raster, component="unknown")

    @pytest.mark.parametrize("spatial_type", ["raster", "point"])
    @pytest.mark.parametrize("chunked", [False, True])
    def test_predict_magnitude__error_predictor_size(self, spatial_type: str, chunked: bool) -> None:
        """Checks an error is raised when a predictor has the wrong shape."""

        # We build a predictor shorter by one than the spatial support
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        chunksizes: int | tuple[int, int] | None
        if spatial_type == "raster":
            like = gu.Raster.from_array(np.ones((2, 3)), transform=from_origin(0, 2, 1, 1), crs=32631)
            predictor = np.ones(5)
            chunksizes = (1, 2) if chunked else None
            if chunked:
                pytest.importorskip("dask")
        else:
            like = gu.PointCloud.from_xyz([0, 1, 2], [0, 1, 2], [1, 1, 1], crs=32631)
            predictor = np.ones(2)
            chunksizes = 2 if chunked else None
            if chunked:
                pytest.importorskip("dask_geopandas")

        # Should fail for both eager/Dask
        with pytest.raises(
            ValueError, match="Spatial predictor 'quality' must be scalar or match every output location"
        ):
            structure.predict_magnitude({"quality": predictor}, like=like, chunksizes=chunksizes)

    @pytest.mark.parametrize("chunked", [False, True])
    def test_predict_magnitude__error_missing_point_column(self, chunked: bool) -> None:
        """Checks an error is raised when a named point predictor column is absent."""

        if chunked:
            pytest.importorskip("dask_geopandas")
        points = gu.PointCloud.from_xyz([0, 1], [0, 1], [1, 1], crs=32631)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        with pytest.raises(ValueError, match="Point cloud predictor column 'absent' does not exist"):
            structure.predict_magnitude({"quality": "absent"}, like=points, chunksizes=1 if chunked else None)

    @pytest.mark.parametrize("spatial_type, chunksizes", [("raster", 3), ("point", (2, 2))])
    def test_predict_magnitude__error_invalid_chunks(self, spatial_type: str, chunksizes: Any) -> None:
        """Checks an error is raised for a chunk size that does not match the spatial input."""

        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        if spatial_type == "raster":
            like = gu.Raster.from_array(np.ones((2, 2)), transform=from_origin(0, 2, 1, 1), crs=32631)
        else:
            like = gu.PointCloud.from_xyz([0, 1], [0, 1], [1, 1], crs=32631)

        with pytest.raises(ValueError, match="chunk size"):
            structure.predict_magnitude(like=like, chunksizes=chunksizes)

    def test_predict_magnitude__error_conflicting_backends(self, tmp_path: Path) -> None:
        """Checks an error is raised when Dask/MP are used together."""

        raster = gu.Raster.from_array(np.ones((2, 2)), transform=from_origin(0, 2, 1, 1), crs=32631)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Check MP+Dask raises an error
        with MpCluster({"nb_workers": 2}) as cluster:
            config = MultiprocConfig(chunks=(1, 1), outfile=str(tmp_path / "magnitude.tif"), cluster=cluster)
            with pytest.raises(ValueError, match="Choose chunksizes for Dask or mp_config for multiprocessing"):
                structure.predict_magnitude(like=raster, chunksizes=(1, 1), mp_config=config)

    def test_predict_magnitude__error_multiproc_with_dask_raster(self, tmp_path: Path) -> None:
        """Checks an error is raised when MP output receives a Dask raster input."""

        dask = pytest.importorskip("dask.array")

        # We create a Dask input
        values = dask.from_array(np.ones((2, 2)), chunks=(1, 1))
        raster = DataArrayRasterAccessor.from_array(values, transform=from_origin(0, 2, 1, 1), crs=32631)
        structure = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Error should be raised here
        with MpCluster({"nb_workers": 2}) as cluster:
            config = MultiprocConfig(chunks=(1, 1), outfile=str(tmp_path / "magnitude.tif"), cluster=cluster)
            with pytest.raises(ValueError, match="Multiprocessing cannot be combined with Dask raster inputs"):
                structure.predict_magnitude(like=raster, mp_config=config)
        assert hasattr(raster.data, "compute")
