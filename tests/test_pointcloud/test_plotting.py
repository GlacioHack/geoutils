"""Test point cloud plotting and opening downsampling."""

from __future__ import annotations

import os
import pathlib
import tempfile

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pytest
from geopandas.testing import assert_geodataframe_equal

import geoutils as gu


def _point_grid(size: int = 10) -> gpd.GeoDataFrame:
    """Create a square point grid with one unique value per row."""

    x = np.tile(np.arange(size), size)
    y = np.repeat(np.arange(size), size)
    return gpd.GeoDataFrame(
        {"value": np.arange(size * size)},
        geometry=gpd.points_from_xy(x, y),
        crs=4326,
    )


class TestPlot:
    """Test module for point plot appearance, deterministic limits, CRS matching, and validation."""

    def test_plot__options_and_save(self) -> None:
        """Checks that established axes, colorbar, styling, and save options remain available."""

        # Create a small point cloud that does not need automatic subsampling
        pointcloud = gu.PointCloud(_point_grid(5), data_column="value")

        # Plot on supplied axes with explicit color limits and return both axes
        ax = plt.subplot(111)
        returned_ax, colorbar_ax = pointcloud.plot(
            cmap="gray",
            vmin=0,
            vmax=20,
            cbar_title="Custom cbar",
            ax=ax,
            return_axes=True,
        )
        assert returned_ax is ax
        assert colorbar_ax is not None
        plt.close()

        # Create new axes and save the resulting figure
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, "test.png")
            pointcloud.plot(ax="new", savefig_fname=filename)
            assert os.path.isfile(filename)
        plt.close()

    def test_plot__point_budget(self) -> None:
        """Checks that a point limit draws the deterministic subsample and keeps the complete source extent."""

        # Create 100 points whose edge coordinates may not appear in a small random sample
        pointcloud = gu.PointCloud(_point_grid(), data_column="value")
        expected = pointcloud.subsample(7, random_state=0)

        # Draw only seven points with the same deterministic sampler
        pointcloud.plot(max_points=7, random_state=0, add_cbar=False)
        ax = plt.gca()
        collection = ax.collections[0]

        assert len(collection.get_offsets()) == 7
        np.testing.assert_array_equal(collection.get_array(), expected.data)
        assert ax.get_xlim()[0] <= pointcloud.bounds.left
        assert ax.get_xlim()[1] >= pointcloud.bounds.right
        assert ax.get_ylim()[0] <= pointcloud.bounds.bottom
        assert ax.get_ylim()[1] >= pointcloud.bounds.top
        plt.close()

    def test_plot__match_reference_crs(self) -> None:
        """Checks that reference matching reprojects only the temporary point sample."""

        # Create points in geographic coordinates and a point cloud that supplies the reference CRS
        pointcloud = gu.PointCloud(_point_grid(3), data_column="value")
        reference = gu.PointCloud(_point_grid(2).to_crs(3857), data_column="value")
        original = pointcloud.ds.geometry.copy()

        # Plot every point in Web Mercator coordinates
        pointcloud.plot(ref_crs=reference, max_points=None, add_cbar=False)
        offsets = np.asarray(plt.gca().collections[0].get_offsets())

        assert pointcloud.crs.to_epsg() == 4326
        assert pointcloud.ds.geometry.equals(original)
        assert offsets[:, 0].max() > 100_000
        plt.close()

    def test_plot__accessor(self) -> None:
        """Checks that the Pandas point cloud accessor exposes the shared plotting implementation."""

        # Use PointCloud construction to attach the main data column metadata to the dataframe
        pointcloud = gu.PointCloud(_point_grid(4), data_column="value")
        dataframe = pointcloud.ds
        dataframe.pc.plot(max_points=5, add_cbar=False)

        assert len(plt.gca().collections[0].get_offsets()) == 5
        plt.close()

    def test_plot__errors(self) -> None:
        """Checks that invalid axes and point limits raise clear errors."""

        pointcloud = gu.PointCloud(_point_grid(3), data_column="value")

        with pytest.raises(ValueError, match="ax must be a matplotlib.axes.Axes instance"):
            pointcloud.plot(ax="wrong_type")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="max_points must be"):
            pointcloud.plot(max_points=0)
        plt.close("all")


class TestPlotChunked:
    """Test module for lazy point plotting, bounded row loading, and eager result equivalence."""

    def test_plot__loading_laziness(self, tmp_path: pathlib.Path) -> None:
        """Checks that plotting materializes only selected rows and keeps the source dataframe lazy."""

        # Write 100 points and reopen them in partitions of 17, including a shorter final partition
        filename = tmp_path / "points.gpkg"
        _point_grid().to_file(filename, index=False)
        lazy = gu.open_pointcloud(str(filename), data_column="value", chunks=17)
        eager = gu.open_pointcloud(str(filename), data_column="value")
        assert lazy.pc._is_dask

        # Plot the same deterministic seven-row sample from both backends
        lazy.pc.plot(max_points=7, random_state=0, add_cbar=False)
        lazy_offsets = np.asarray(plt.gca().collections[0].get_offsets())
        lazy_values = np.asarray(plt.gca().collections[0].get_array())
        plt.close()
        eager.pc.plot(max_points=7, random_state=0, add_cbar=False)
        eager_offsets = np.asarray(plt.gca().collections[0].get_offsets())
        eager_values = np.asarray(plt.gca().collections[0].get_array())

        # The input remains partitioned and produces exactly the eager sample
        assert lazy.pc._is_dask
        np.testing.assert_array_equal(lazy_offsets, eager_offsets)
        np.testing.assert_array_equal(lazy_values, eager_values)
        plt.close()


class TestOpenDownsample:
    """Test module for eager, lazy, and file-backed point cloud opening downsampling."""

    def test_pointcloud__downsample(self) -> None:
        """Checks that PointCloud opening keeps the requested deterministic fraction of loaded rows."""

        # Downsample a loaded dataframe by a factor of four
        dataframe = _point_grid()
        pointcloud = gu.PointCloud(dataframe, data_column="value", downsample=4)
        expected = gu.PointCloud(dataframe, data_column="value").subsample(25, random_state=0)

        assert pointcloud.point_count == 25
        assert_geodataframe_equal(pointcloud.ds.reset_index(drop=True), expected.ds.reset_index(drop=True))

    def test_pointcloud__downsample_lazy_file(self, tmp_path: pathlib.Path) -> None:
        """Checks that a file-backed PointCloud reports and loads its reduced row count."""

        # Keep the file unopened while exposing the eventual sample size from metadata
        filename = tmp_path / "points.gpkg"
        _point_grid().to_file(filename, index=False)
        pointcloud = gu.PointCloud(filename, data_column="value", downsample=4)
        assert not pointcloud.is_loaded
        assert pointcloud.point_count == 25

        # Loading applies the deterministic sample exactly once
        pointcloud.load()
        assert pointcloud.is_loaded
        assert pointcloud.point_count == 25

    def test_open_pointcloud__downsample_chunked(self, tmp_path: pathlib.Path) -> None:
        """Checks that chunked opening returns the same deterministic sample as eager opening and stays lazy."""

        # Open the same file eagerly and in uneven Dask partitions
        filename = tmp_path / "points.gpkg"
        _point_grid().to_file(filename, index=False)
        eager = gu.open_pointcloud(str(filename), data_column="value", downsample=4)
        lazy = gu.open_pointcloud(str(filename), data_column="value", chunks=17, downsample=4)

        # Dask keeps only the selected rows in its graph and matches the eager top-k sample
        assert lazy.pc._is_dask
        assert lazy.pc.point_count == 25
        assert_geodataframe_equal(
            lazy.compute().reset_index(drop=True),
            eager.reset_index(drop=True),
        )

    @pytest.mark.parametrize("downsample", [0, -1, np.inf, "wrong"])
    def test_open_pointcloud__error_downsample(self, downsample: object) -> None:
        """Checks that invalid opening downsampling factors raise clear errors."""

        with pytest.raises((TypeError, ValueError), match="downsample must be"):
            gu.PointCloud(_point_grid(2), data_column="value", downsample=downsample)  # type: ignore[arg-type]
