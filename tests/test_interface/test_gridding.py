"""Test point-cloud gridding values and consistency across calculation backends."""

from importlib.util import find_spec
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio
import xarray as xr
from shapely import geometry

import geoutils as gu
from geoutils import PointCloud, Raster
from geoutils.interface.gridding import GriddingMethod, _grid_pointcloud
from geoutils.multiproc import MultiprocConfig


class TestPointCloud:
    """Test interpolation and neighborhood methods used to grid point clouds."""

    def test_grid_pc(self) -> None:
        """Test point cloud gridding."""

        # 1/ Check gridding interpolation falls back exactly on original raster

        # Create a point cloud from interpolating a grid, so we can compare back after to check consistency
        rng = np.random.default_rng(42)
        shape = (10, 12)
        rst_arr = np.linspace(0, 10, int(np.prod(shape))).reshape(*shape)
        transform = rio.transform.from_origin(0, shape[0] - 1, 1, 1)
        rst = Raster.from_array(rst_arr, transform=transform, crs=4326, nodata=100)

        # Generate random coordinates to interpolate, to create an irregular point cloud
        points = rng.integers(low=1, high=shape[0] - 1, size=(100, 2)) + rng.normal(0, 0.15, size=(100, 2))
        b1_value = rst.interp_points((points[:, 0], points[:, 1]), as_array=True)
        pc = gpd.GeoDataFrame(data={"b1": b1_value}, geometry=gpd.points_from_xy(x=points[:, 0], y=points[:, 1]))
        grid_coords = rst.coords(grid=False)

        # Grid the point cloud
        gridded_pc, output_transform = _grid_pointcloud(pc, grid_coords=grid_coords, data_column_name="b1")

        # Compare back to raster, all should be very close (but not exact, some info is lost due to interpolations)
        valids = np.isfinite(gridded_pc)
        assert np.allclose(gridded_pc[valids], rst.data.data[valids], rtol=10e-5)
        # And the transform exactly the same
        assert output_transform == transform

        # 2/ Check the propagation of nodata values

        # 2.1/ Grid points outside the convex hull of all points should always be nodata

        # We convert the full raster to a point cloud, keeping all cells even nodata
        rst_pc = rst.to_pointcloud(skip_nodata=False).ds

        # We define a multi-point geometry from the individual points, and compute its convex hull
        poly = geometry.MultiPoint([[p.x, p.y] for p in pc.geometry])
        chull = poly.convex_hull

        # We compute the index of grid cells intersecting the convex hull
        ind_inters_convhull = rst_pc.intersects(chull)

        # We get corresponding 1D indexes for gridded output
        i, j = rst.xy2ij(x=rst_pc.geometry.x.values, y=rst_pc.geometry.y.values)

        # Check all values outside convex hull are NaNs
        assert all(~np.isfinite(gridded_pc[i[~ind_inters_convhull], j[~ind_inters_convhull]]))

        # 2.2/ For the rest of the points, data should be valid only if a point exists within 1 pixel of their
        # coordinate, that is the closest rounded number
        # TODO: Replace by check with distance, because some pixel not rounded can also be at less than 1 from a point

        # Compute min distance to irregular point cloud for each grid point
        list_min_dist = []
        for p in rst_pc.geometry:
            min_dist = np.min(np.sqrt((p.x - pc.geometry.x.values) ** 2 + (p.y - pc.geometry.y.values) ** 2))
            list_min_dist.append(min_dist)

        ind_close = np.array(list_min_dist) <= 1
        # We get the indexes for these coordinates
        iround, jround = rst.xy2ij(x=rst_pc.geometry.x.values[ind_close], y=rst_pc.geometry.y.values[ind_close])

        # Keep only indexes in the convex hull
        indexes_close = [(iround[k], jround[k]) for k in range(len(iround))]
        indexes_chull = [(i[k], j[k]) for k in range(len(i)) if ind_inters_convhull[k]]
        close_in_chull = [tup for tup in indexes_close if tup in indexes_chull]
        iclosechull, jclosehull = list(zip(*close_in_chull))

        # All values close to pixel in the convex hull should be valid
        assert all(np.isfinite(gridded_pc[iclosechull, jclosehull]))

        # Other values in the convex hull should not be
        far_in_chull = [tup for tup in indexes_chull if tup not in indexes_close]
        ifarchull, jfarchull = list(zip(*far_in_chull))

        assert all(~np.isfinite(gridded_pc[ifarchull, jfarchull]))

        # Check for a different distance value
        gridded_pc, output_transform = _grid_pointcloud(
            pc, grid_coords=grid_coords, dist_nodata_pixel=0.5, data_column_name="b1"
        )
        ind_close = np.array(list_min_dist) <= 0.5

        # We get the indexes for these coordinates
        iround, jround = rst.xy2ij(x=rst_pc.geometry.x.values[ind_close], y=rst_pc.geometry.y.values[ind_close])

        # Keep only indexes in the convex hull
        indexes_close = [(iround[k], jround[k]) for k in range(len(iround))]
        indexes_chull = [(i[k], j[k]) for k in range(len(i)) if ind_inters_convhull[k]]
        close_in_chull = [tup for tup in indexes_close if tup in indexes_chull]
        iclosechull, jclosehull = list(zip(*close_in_chull))

        # All values close  pixel in the convex hull should be valid
        assert all(np.isfinite(gridded_pc[iclosechull, jclosehull]))

        # Other values in the convex hull should not be
        far_in_chull = [tup for tup in indexes_chull if tup not in indexes_close]
        ifarchull, jfarchull = list(zip(*far_in_chull))

        assert all(~np.isfinite(gridded_pc[ifarchull, jfarchull]))

        # Infinite support skips distance filtering but must match a sufficiently large finite cutoff
        finite_support, _ = _grid_pointcloud(
            pc,
            grid_coords=grid_coords,
            data_column_name="b1",
            resampling="nearest",
            dist_nodata_pixel=1e9,
        )
        infinite_support, _ = _grid_pointcloud(
            pc,
            grid_coords=grid_coords,
            data_column_name="b1",
            resampling="nearest",
            dist_nodata_pixel=float("inf"),
        )
        assert np.array_equal(finite_support, infinite_support, equal_nan=True)

        # 3/ Errors
        with pytest.raises(TypeError, match="Input grid coordinates must be 1D arrays.*"):
            Raster.from_pointcloud_regular(pc, grid_coords=(1, "lol"))  # type: ignore
        with pytest.raises(ValueError, match="Grid coordinates must be regular*"):
            grid_coords[0][0] += 1
            Raster.from_pointcloud_regular(pc, grid_coords=grid_coords)  # type: ignore

    @pytest.mark.parametrize("resampling", ["idw", "mean"])
    def test_grid_pc__circular_neighborhood(self, resampling: GriddingMethod) -> None:
        """Check IDW and moving means on points with an exact analytical result."""

        # Two constant-valued columns place an equal pair of neighbors around the central column
        pc = gpd.GeoDataFrame(
            data={"z": [0.0, 10.0, 0.0, 10.0]},
            geometry=gpd.points_from_xy(x=[0.0, 2.0, 0.0, 2.0], y=[0.0, 0.0, 1.0, 1.0]),
        )
        grid_coords = (np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]))

        # A radius just over one pixel reaches both same-row neighbors at the center
        result, _ = _grid_pointcloud(
            pc,
            grid_coords=grid_coords,
            data_column_name="z",
            resampling=resampling,
            dist_nodata_pixel=1.1,
        )
        expected = np.array([[0.0, 5.0, 10.0], [0.0, 5.0, 10.0]])
        assert np.allclose(result, expected)

    @pytest.mark.parametrize("resampling", ["nearest", "linear"])
    def test_grid_pc__nodata_policies(self, resampling: GriddingMethod) -> None:
        """Apply the same default, ignored and propagated nodata rules as raster interpolation."""

        # A regular point grid has one invalid value that surrounding finite values can replace
        x, y = np.meshgrid(np.arange(3, dtype=float), np.arange(3, dtype=float))
        values = np.arange(9, dtype=float)
        values[4] = np.nan
        pc = gpd.GeoDataFrame(data={"z": values}, geometry=gpd.points_from_xy(x=x.ravel(), y=y.ravel()))
        grid_coords = (np.arange(3, dtype=float), np.arange(3, dtype=float))

        # GDAL gridding and the explicit ignore rule both omit invalid point values
        default, _ = _grid_pointcloud(
            pc,
            grid_coords=grid_coords,
            data_column_name="z",
            resampling=resampling,
        )
        ignored, _ = _grid_pointcloud(
            pc,
            grid_coords=grid_coords,
            data_column_name="z",
            resampling=resampling,
            nodata_propagation="ignore",
        )
        propagated, _ = _grid_pointcloud(
            pc,
            grid_coords=grid_coords,
            data_column_name="z",
            resampling=resampling,
            nodata_propagation="propagate",
        )
        assert np.array_equal(default, ignored, equal_nan=True)
        assert np.isfinite(default[1, 1])
        assert np.isnan(propagated[1, 1])

    def test_grid_pc__circular_nodata_propagation(self) -> None:
        """Propagate an invalid point through the complete circular support when requested."""

        # The finite endpoints give every central neighborhood a result when invalid values are ignored
        pc = gpd.GeoDataFrame(
            data={"z": [2.0, np.nan, 8.0]},
            geometry=gpd.points_from_xy(x=[0.0, 1.0, 2.0], y=[0.0, 0.0, 0.0]),
        )
        grid_coords = (np.arange(4, dtype=float), np.array([0.0]))
        default, _ = _grid_pointcloud(
            pc,
            grid_coords=grid_coords,
            grid_res=(1.0, 1.0),
            data_column_name="z",
            resampling="mean",
            dist_nodata_pixel=1.1,
        )
        propagated, _ = _grid_pointcloud(
            pc,
            grid_coords=grid_coords,
            grid_res=(1.0, 1.0),
            data_column_name="z",
            resampling="mean",
            dist_nodata_pixel=1.1,
            nodata_propagation="propagate",
        )

        # The last cell lies outside the invalid point support and therefore remains unchanged
        assert np.all(np.isfinite(default[0, :3]))
        assert np.all(np.isnan(propagated[0, :3]))
        assert propagated[0, 3] == default[0, 3]

    @pytest.mark.parametrize(
        ("resampling", "expected_center"),
        [
            ("mean", 5.0),
            ("average", 5.0),
            ("minimum", 2.0),
            ("min", 2.0),
            ("maximum", 8.0),
            ("max", 8.0),
            ("range", 6.0),
            ("count", 2.0),
            ("stdev", 3.0),
            ("average_distance", 1.0),
            ("average_distance_pts", 2.0),
        ],
    )
    def test_grid_pc__circular_statistics(self, resampling: GriddingMethod, expected_center: float) -> None:
        """Check circular statistics, aliases and the shared handling of invalid values."""

        # Two finite points surround the central cell while invalid values cannot contribute
        pc = gpd.GeoDataFrame(
            data={"z": [2.0, 8.0, np.nan, 20.0]},
            geometry=gpd.points_from_xy(x=[0.0, 2.0, 1.0, np.nan], y=[0.0, 0.0, 0.0, 0.0]),
        )
        grid_coords = (np.arange(5, dtype=float), np.array([0.0]))

        # Cells with neighbors use only finite values and cells outside support remain NaN
        result, _ = _grid_pointcloud(
            pc,
            grid_coords=grid_coords,
            grid_res=(1.0, 1.0),
            data_column_name="z",
            resampling=resampling,
            dist_nodata_pixel=1.1,
        )
        assert result[0, 1] == pytest.approx(expected_center)
        assert np.isnan(result[0, 4])

    def test_grid_pc__minimum_points(self) -> None:
        """Check that circular outputs need the requested number of finite points."""

        # Only the central cell reaches both points inside its circular support
        pc = gpd.GeoDataFrame(
            data={"z": [2.0, 8.0]},
            geometry=gpd.points_from_xy(x=[0.0, 2.0], y=[0.0, 0.0]),
        )
        result, _ = _grid_pointcloud(
            pc,
            grid_coords=(np.array([0.0, 1.0, 2.0]), np.array([0.0])),
            grid_res=(1.0, 1.0),
            data_column_name="z",
            resampling="mean",
            dist_nodata_pixel=1.1,
            min_points=2,
        )
        assert np.array_equal(result, np.array([[np.nan, 5.0, np.nan]]), equal_nan=True)

    def test_grid_pc__idw_distance_power_and_exact_points(self) -> None:
        """Check that IDW follows its distance exponent and preserves exact source values."""

        # The first output cell coincides with a point while the middle cell has unequal distances
        pc = gpd.GeoDataFrame(
            data={"z": [0.0, 10.0]},
            geometry=gpd.points_from_xy(x=[0.0, 3.0], y=[0.0, 0.0]),
        )
        grid_coords = (np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 1.0]))

        # Squared inverse distances give weights of one and one quarter at the inner columns
        result, _ = _grid_pointcloud(
            pc,
            grid_coords=grid_coords,
            data_column_name="z",
            resampling="idw",
            dist_nodata_pixel=2.1,
            distance_power=2,
        )
        assert np.allclose(result[1], [0.0, 2.0, 8.0, 10.0])

    @pytest.mark.parametrize(
        "resampling",
        ["nearest", "idw", "mean", "minimum", "maximum", "range", "count", "stdev", "average_distance"],
    )
    def test_grid_pc__engine(self, resampling: GriddingMethod) -> None:
        """Check that the SciPy and Numba engines give the same gridded values."""

        pytest.importorskip("numba")

        # Uneven point values and positions exercise distance choices and every accumulation
        pc = gpd.GeoDataFrame(
            data={"z": [1.0, 4.0, 8.0]},
            geometry=gpd.points_from_xy(x=[0.0, 1.2, 3.0], y=[0.0, 1.0, 0.0]),
        )
        grid_coords = (np.arange(4, dtype=float), np.arange(2, dtype=float))
        scipy_result, _ = _grid_pointcloud(
            pc,
            grid_coords=grid_coords,
            data_column_name="z",
            resampling=resampling,
            dist_nodata_pixel=2,
            engine="scipy",
        )

        # The explicit Numba engine follows the same interface as elsewhere in GeoUtils and xDEM
        numba_result, _ = _grid_pointcloud(
            pc,
            grid_coords=grid_coords,
            data_column_name="z",
            resampling=resampling,
            dist_nodata_pixel=2,
            engine="numba",
        )
        assert np.allclose(scipy_result, numba_result, equal_nan=True)

    @pytest.mark.parametrize("resampling", ["linear", "cubic", "average_distance_pts"])
    def test_grid_pc__numba_unsupported_method(self, resampling: GriddingMethod) -> None:
        """Raise a clear error for gridding methods without a Numba implementation."""

        pc = gpd.GeoDataFrame(data={"z": [1.0]}, geometry=gpd.points_from_xy(x=[0.0], y=[0.0]))
        grid_coords = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))

        with pytest.raises(ValueError, match="Numba gridding engine does not support"):
            _grid_pointcloud(
                pc,
                grid_coords=grid_coords,
                data_column_name="z",
                resampling=resampling,
                engine="numba",
            )

    @pytest.mark.skipif(find_spec("numba") is not None, reason="Only runs if numba is missing.")
    def test_grid_pc__numba_missing_dependency(self) -> None:
        """Raise the standard optional-dependency error when Numba is unavailable."""

        pc = gpd.GeoDataFrame(data={"z": [1.0]}, geometry=gpd.points_from_xy(x=[0.0], y=[0.0]))
        with pytest.raises(ImportError, match="Optional dependency 'numba' required"):
            _grid_pointcloud(
                pc,
                grid_coords=(np.array([0.0, 1.0]), np.array([0.0, 1.0])),
                data_column_name="z",
                resampling="nearest",
                engine="numba",
            )

    def test_grid_pc__neighborhood_errors(self) -> None:
        """Check explicit validation for unsupported neighborhood definitions."""

        pc = gpd.GeoDataFrame(data={"z": [1.0]}, geometry=gpd.points_from_xy(x=[0.0], y=[0.0]))
        grid_coords = (np.array([0.0, 1.0]), np.array([0.0, 1.0]))

        # Local aggregation cannot have infinite support without building all point-cell pairs
        with pytest.raises(ValueError, match="require a finite dist_nodata_pixel"):
            _grid_pointcloud(
                pc,
                grid_coords=grid_coords,
                data_column_name="z",
                resampling="mean",
                dist_nodata_pixel=float("inf"),
            )
        with pytest.raises(ValueError, match="distance_power must be finite and strictly positive"):
            _grid_pointcloud(
                pc,
                grid_coords=grid_coords,
                data_column_name="z",
                resampling="idw",
                distance_power=0,
            )
        with pytest.raises(ValueError, match="min_points.*non-negative integer"):
            _grid_pointcloud(
                pc,
                grid_coords=grid_coords,
                data_column_name="z",
                resampling="count",
                min_points=-1,
            )
        with pytest.raises(ValueError, match="nodata_propagation must be one of"):
            _grid_pointcloud(
                pc,
                grid_coords=grid_coords,
                data_column_name="z",
                nodata_propagation="invalid",  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="engine.*either 'scipy' or 'numba'"):
            _grid_pointcloud(
                pc,
                grid_coords=grid_coords,
                data_column_name="z",
                engine="invalid",  # type: ignore[arg-type]
            )


class TestGridChunked:
    """Compare gridding outputs and loading across eager, Dask and Multiprocessing backends."""

    # Use a regular point grid so every interpolation method has enough local support
    x, y = np.meshgrid(np.arange(3, dtype=float), np.arange(3, dtype=float))
    points = gpd.GeoDataFrame(
        data={"z": np.arange(9, dtype=float)},
        geometry=gpd.points_from_xy(x=x.ravel(), y=y.ravel()),
        crs=32610,
    )
    grid_coords = (np.arange(3, dtype=float), np.arange(3, dtype=float))

    @pytest.mark.parametrize(
        "resampling",
        [
            "nearest",
            "linear",
            "cubic",
            "idw",
            "mean",
            "minimum",
            "maximum",
            "range",
            "count",
            "stdev",
            "average_distance",
            "average_distance_pts",
        ],
    )
    def test_grid__chunked_backends_equal(self, resampling: GriddingMethod, tmp_path: Path) -> None:
        """
        Test that grid returns exactly the same output for:
         - PointCloud and the Pandas accessor in memory,
         - Dask through the Pandas accessor with lazy input and output,
         - Multiprocessing through PointCloud with lazy input and output.

        The Dask and Multiprocessing inputs must remain unloaded after their results are read.
        """

        pytest.importorskip("dask_geopandas")
        import dask.array as da

        # Store one point source for both backends that read partitions from disk
        point_file = tmp_path / "points.gpkg"
        self.points.to_file(point_file)

        # 1/ Prepare the same point cloud through every public interface
        pointcloud = PointCloud(self.points, data_column="z")
        point_accessor = self.points.copy()
        point_accessor.pc.set_data_column("z")
        dask_points = gu.open_pointcloud(str(point_file), data_column="z", chunks=3)
        multiproc_points = PointCloud(point_file, data_column="z")

        assert pointcloud.is_loaded
        assert point_accessor.pc.is_loaded
        assert not dask_points.pc.is_loaded
        assert not multiproc_points.is_loaded

        # 2/ Grid the complete source eagerly and split the other outputs into rectangular chunks
        kwargs = {
            "grid_coords": self.grid_coords,
            "resampling": resampling,
            "dist_nodata_pixel": 2,
        }
        expected = pointcloud.grid(**kwargs)
        accessor_output = point_accessor.pc.grid(**kwargs)
        dask_output = dask_points.pc.grid(**kwargs, chunksizes=(2, 1))
        multiproc_output = multiproc_points.grid(
            **kwargs,
            mp_config=MultiprocConfig(chunks=(2, 1), outfile=str(tmp_path / "grid-multiproc.tif")),
        )

        # 3/ Check output types and loading before evaluating the chunked results
        assert isinstance(expected, Raster)
        assert expected.is_loaded
        assert isinstance(accessor_output, xr.DataArray)
        assert accessor_output._in_memory
        assert isinstance(dask_output, xr.DataArray)
        assert isinstance(dask_output.data, da.Array)
        assert not dask_output._in_memory
        assert dask_output.data.chunks == ((2, 1), (1, 1, 1))
        assert isinstance(multiproc_output, Raster)
        assert not multiproc_output.is_loaded

        # 4/ Read the results and require exact values, masks and georeferencing
        computed_dask = dask_output.compute()
        multiproc_output.load()
        assert expected.raster_equal(accessor_output, warn_failure_reason=True, strict_masked=False)
        assert expected.raster_equal(computed_dask, warn_failure_reason=True, strict_masked=False)
        assert expected.raster_equal(multiproc_output, warn_failure_reason=True, strict_masked=False)

        # 5/ Computing an output must not replace either lazy source with in-memory data
        assert not dask_points.pc.is_loaded
        assert not multiproc_points.is_loaded
        assert not dask_output._in_memory
        assert multiproc_output.is_loaded

    @pytest.mark.parametrize("point_dask", [False, True], ids=["point-eager", "point-dask"])
    @pytest.mark.parametrize("raster_dask", [False, True], ids=["raster-eager", "raster-dask"])
    def test_grid__point_raster_input_combinations(
        self,
        point_dask: bool,
        raster_dask: bool,
        tmp_path: Path,
    ) -> None:
        """Grid every combination of eager and Dask point-cloud and raster reference inputs."""

        pytest.importorskip("dask_geopandas")
        import dask.array as da

        # Write both inputs so their Dask variants use the same values and georeferencing
        point_file = tmp_path / "points.gpkg"
        self.points.to_file(point_file)
        reference = Raster.from_array(
            np.zeros((3, 3), dtype=np.uint8),
            transform=rio.transform.from_origin(-0.5, 2.5, 1, 1),
            crs=self.points.crs,
        )
        raster_file = tmp_path / "reference.tif"
        reference.to_file(raster_file)

        # Select each input independently to cover all four eager and Dask combinations
        points = (
            gu.open_pointcloud(str(point_file), data_column="z", chunks=3)
            if point_dask
            else PointCloud(self.points, data_column="z")
        )
        raster = gu.open_raster(str(raster_file), chunks={"x": 2, "y": 2}) if raster_dask else reference
        expected = PointCloud(self.points, data_column="z").grid(
            ref=reference,
            resampling="nearest",
            dist_nodata_pixel=2,
        )

        # The point-cloud backend controls whether the output itself is eager or Dask
        output = (
            points.pc.grid(ref=raster, resampling="nearest", dist_nodata_pixel=2)
            if point_dask
            else points.grid(
                ref=raster,
                resampling="nearest",
                dist_nodata_pixel=2,
            )
        )
        if point_dask:
            assert isinstance(output, xr.DataArray)
            assert isinstance(output.data, da.Array)
            assert not output._in_memory
            computed_output = output.compute()
            assert not points.pc.is_loaded
        else:
            assert isinstance(output, Raster)
            assert output.is_loaded
            computed_output = output

        # A Dask reference supplies only grid metadata and chunks, and must stay lazy
        if raster_dask:
            assert isinstance(raster.data, da.Array)
            assert not raster._in_memory
            if point_dask:
                assert output.data.chunks == raster.data.chunks

        assert expected.raster_equal(computed_output, warn_failure_reason=True, strict_masked=False)

    @pytest.mark.parametrize("resampling", ["nearest", "mean"])
    def test_grid__numba_chunked_backends(self, resampling: GriddingMethod, tmp_path: Path) -> None:
        """Keep the Numba calculation engine identical across eager, Dask and Multiprocessing backends."""

        pytest.importorskip("numba")
        pytest.importorskip("dask_geopandas")

        # Store one point source so both chunked backends can select local points
        point_file = tmp_path / "points.gpkg"
        self.points.to_file(point_file)
        kwargs = {
            "grid_coords": self.grid_coords,
            "resampling": resampling,
            "dist_nodata_pixel": 2,
            "engine": "numba",
        }

        # Compare both chunked outputs with one complete eager calculation
        expected = PointCloud(self.points, data_column="z").grid(**kwargs)
        dask_points = gu.open_pointcloud(str(point_file), data_column="z", chunks=3)
        dask_output = dask_points.pc.grid(**kwargs, chunksizes=(2, 1))
        multiproc_points = PointCloud(point_file, data_column="z")
        multiproc_output = multiproc_points.grid(
            **kwargs,
            mp_config=MultiprocConfig(chunks=(2, 1), outfile=str(tmp_path / "grid-numba.tif")),
        )
        assert expected.raster_equal(dask_output.compute(), warn_failure_reason=True, strict_masked=False)
        assert expected.raster_equal(multiproc_output, warn_failure_reason=True, strict_masked=False)
        assert not dask_points.pc.is_loaded
        assert not multiproc_points.is_loaded

    def test_grid__nodata_propagation_chunked_backends(self, tmp_path: Path) -> None:
        """Keep propagated invalid points identical across eager, Dask and Multiprocessing gridding."""

        pytest.importorskip("dask_geopandas")

        # Add one invalid center to exercise support across output chunk boundaries
        points = self.points.copy()
        points.loc[4, "z"] = np.nan
        point_file = tmp_path / "points-nodata.gpkg"
        points.to_file(point_file)
        kwargs = {
            "grid_coords": self.grid_coords,
            "resampling": "mean",
            "dist_nodata_pixel": 1.1,
            "nodata_propagation": "propagate",
        }

        # Compare the same nodata rule before and after splitting either input or output
        expected = PointCloud(points, data_column="z").grid(**kwargs)
        dask_points = gu.open_pointcloud(str(point_file), data_column="z", chunks=3)
        dask_output = dask_points.pc.grid(**kwargs, chunksizes=(2, 2))
        multiproc_points = PointCloud(point_file, data_column="z")
        multiproc_output = multiproc_points.grid(
            **kwargs,
            mp_config=MultiprocConfig(chunks=(2, 2), outfile=str(tmp_path / "grid-nodata.tif")),
        )
        assert expected.raster_equal(dask_output.compute(), warn_failure_reason=True, strict_masked=False)
        assert expected.raster_equal(multiproc_output, warn_failure_reason=True, strict_masked=False)
        assert not dask_points.pc.is_loaded
        assert not multiproc_points.is_loaded

    def test_grid__dask_multiprocessing_error(self, tmp_path: Path) -> None:
        """Reject two schedulers for one gridding operation before evaluating point partitions."""

        pytest.importorskip("dask_geopandas")

        # A Dask point source already owns task scheduling and cannot use Multiprocessing
        point_file = tmp_path / "points.gpkg"
        self.points.to_file(point_file)
        points = gu.open_pointcloud(str(point_file), data_column="z", chunks=3)
        with pytest.raises(ValueError, match="Cannot use Multiprocessing and Dask simultaneously"):
            points.pc.grid(
                grid_coords=self.grid_coords,
                mp_config=MultiprocConfig(chunks=(2, 2), outfile=str(tmp_path / "grid-error.tif")),
            )
        assert not points.pc.is_loaded
