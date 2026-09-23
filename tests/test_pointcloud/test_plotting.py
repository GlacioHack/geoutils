"""Test point cloud plotting and opening downsampling."""

from __future__ import annotations

import os
import pathlib
import tempfile
from importlib.util import find_spec

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pytest
from affine import Affine

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

    def test_plot__boolean_colorbar(self) -> None:
        """Checks that a boolean point mask uses a labeled continuous colorbar without changing its data."""

        # Create a point mask whose two boolean values would otherwise make GeoPandas select a categorical legend
        dataframe = gpd.GeoDataFrame(
            {"mask": [False, True]},
            geometry=gpd.points_from_xy([0, 1], [0, 1]),
            crs=32632,
        )
        pointcloud = gu.PointCloud(dataframe, data_column="mask")

        # Plot the mask through the normal point cloud colorbar path
        _, colorbar_ax = pointcloud.plot(cbar_title="Intersects vector", return_axes=True)

        assert colorbar_ax is not None
        assert colorbar_ax.get_ylabel() == "Intersects vector"
        assert pointcloud.data.dtype == bool
        plt.close()

    def test_plot__fixed_color(self) -> None:
        """Checks that an explicit color overrides the default data column without creating a colorbar."""

        # Create a point cloud whose main data column would otherwise control marker colors
        pointcloud = gu.PointCloud(_point_grid(3), data_column="value")

        # Draw every point in one color and check that no numeric values or colorbar were attached
        _, colorbar_ax = pointcloud.plot(color="darkred", max_points=None, return_axes=True)
        collection = plt.gca().collections[0]

        assert colorbar_ax is None
        assert collection.get_array() is None
        np.testing.assert_allclose(collection.get_facecolor()[0, :3], (0.5451, 0.0, 0.0), atol=1e-4)
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

    def test_plot__geographic_colorbar_placement(self) -> None:
        """Checks that a plot in geographic CRS keeps its colorbar properly positioned/sized."""

        # Create a high-latitude that strongly adjusts geographic map aspect
        longitudes = np.linspace(15.2, 16.3, 100)
        latitudes = np.linspace(77.95, 78.18, 100)
        dataframe = gpd.GeoDataFrame(
            {"value": np.arange(100)},
            geometry=gpd.points_from_xy(longitudes, latitudes),
            crs=4326,
        )
        pointcloud = gu.PointCloud(dataframe, data_column="value")

        # Draw the figure before comparing the final map and colorbar positions
        ax, colorbar_ax = pointcloud.plot(return_axes=True)
        assert colorbar_ax is not None
        ax.figure.canvas.draw()
        map_position = ax.get_position()
        colorbar_position = colorbar_ax.get_position()
        gap = colorbar_position.x0 - map_position.x1

        # Check that position/shape is correct
        assert 0 <= gap < map_position.width / 2
        assert colorbar_position.height == pytest.approx(map_position.height)
        plt.close(ax.figure)

    def test_plot__reference_panel_size(self) -> None:
        """Checks that a reference-matched point plot has the same map and colorbar size as a raster plot."""

        # Create a wide reference that makes equal-aspect axes shorter than their original subplot slots
        reference = gu.Raster.from_array(np.ones((10, 100)), Affine(1, 0, 0, 0, -1, 10), 3857)
        dataframe = gpd.GeoDataFrame(
            {"value": [0, 1]},
            geometry=gpd.points_from_xy([0, 100], [0, 1]),
            crs=3857,
        )
        pointcloud = gu.PointCloud(dataframe, data_column="value")

        # Plot the same extent in paired panels whose titles occupy one and two lines
        fig, axes = plt.subplots(1, 2)
        axes[0].set_title("Raster")
        raster_ax, raster_colorbar_ax = reference.plot(ax=axes[0], max_pixels=None, return_axes=True)
        axes[1].set_title("Reduced\npoint cloud")
        point_ax, point_colorbar_ax = pointcloud.plot(
            ref=reference,
            ax=axes[1],
            max_points=None,
            return_axes=True,
        )
        assert raster_colorbar_ax is not None
        assert point_colorbar_ax is not None
        fig.tight_layout()
        fig.canvas.draw()

        # Matching data bounds and non-resizing colorbars should give both panels the same size
        raster_position = raster_ax.get_position()
        point_position = point_ax.get_position()
        assert point_position.y0 == pytest.approx(raster_position.y0)
        assert point_position.width == pytest.approx(raster_position.width)
        assert point_position.height == pytest.approx(raster_position.height)
        assert point_colorbar_ax.get_position().width == pytest.approx(raster_colorbar_ax.get_position().width)
        assert point_colorbar_ax.get_position().height == pytest.approx(raster_colorbar_ax.get_position().height)

        plt.close(fig)

    def test_plot__projected_tick_spacing(self) -> None:
        """Checks that long projected coordinates do not overlap on a narrow plot."""

        # Create one narrow panel with six-digit easting labels like those in documentation figures
        dataframe = gpd.GeoDataFrame(
            {"value": [0, 1]},
            geometry=gpd.points_from_xy([470_000, 510_000], [3_080_000, 3_120_000]),
            crs=32632,
        )
        pointcloud = gu.PointCloud(dataframe, data_column="value")
        fig, axes = plt.subplots(1, 3, figsize=(4, 3))

        # Plot in the first panel and compare the rendered bounds of neighboring labels
        pointcloud.plot(ax=axes[0], max_points=None)
        fig.canvas.draw()
        labels = [label for label in axes[0].get_xticklabels() if label.get_visible() and label.get_text()]
        label_boxes = [label.get_window_extent() for label in labels]

        assert all(not first.overlaps(second) for first, second in zip(label_boxes[:-1], label_boxes[1:]))
        plt.close(fig)

    def test_plot__match_reference_crs(self) -> None:
        """Checks that reference matching reprojects the point sample CRS and crops to bounds."""

        # Create points in geographic coordinates and a point cloud that supplies the reference CRS
        pointcloud = gu.PointCloud(_point_grid(3), data_column="value")
        reference = gu.PointCloud(_point_grid(2).to_crs(3857), data_column="value")
        original = pointcloud.ds.geometry.copy()

        # Plot every point in Web Mercator coordinates
        pointcloud.plot(ref=reference, max_points=None, add_cbar=False)
        ax = plt.gca()
        offsets = np.asarray(ax.collections[0].get_offsets())

        # Check input and plot are in the expected CRS and bounds
        assert pointcloud.crs.to_epsg() == 4326
        assert pointcloud.ds.geometry.equals(original)
        assert offsets[:, 0].max() > 100_000
        assert ax.get_xlim() == pytest.approx((reference.bounds.left, reference.bounds.right))
        assert ax.get_ylim() == pytest.approx((reference.bounds.bottom, reference.bounds.top))
        plt.close()

    def test_plot__deprecated_ref_crs(self) -> None:
        """Checks that the old ref_crs argument warns and uses only the reference CRS."""

        # Create a reference in Web Mercator that covers less area than the source
        pointcloud = gu.PointCloud(_point_grid(3), data_column="value")
        reference = gu.PointCloud(_point_grid(2).to_crs(3857), data_column="value")

        # Check that ref_crs changes the CRS without limiting the plotted area
        with pytest.warns(DeprecationWarning, match="Argument 'ref_crs' is deprecated"):
            pointcloud.plot(ref_crs=reference, max_points=None, add_cbar=False)
        ax = plt.gca()
        offsets = np.asarray(ax.collections[0].get_offsets())

        assert offsets[:, 0].max() > reference.bounds.right
        assert ax.get_xlim()[1] > reference.bounds.right
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


@pytest.mark.skipif(find_spec("dask_geopandas") is None, reason="Only runs if dask_geopandas is installed.")
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
