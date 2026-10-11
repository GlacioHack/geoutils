"""Test vector plotting layout and reference matching."""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pytest
from affine import Affine

import geoutils as gu


class TestPlot:
    """Test module for vector colorbar placement and reference axis limits."""

    def test_plot__geographic_colorbar_placement(self) -> None:
        """Checks that a geographic vector plot keeps its colorbar next to axes adjusted for map aspect."""

        # Create the narrow high-latitude extent that strongly adjusts GeoPandas' geographic map aspect
        longitudes = np.linspace(15.2, 16.3, 100)
        latitudes = np.linspace(77.95, 78.18, 100)
        dataframe = gpd.GeoDataFrame(
            {"value": np.arange(100)},
            geometry=gpd.points_from_xy(longitudes, latitudes),
            crs=4326,
        )
        vector = gu.Vector(dataframe)

        # Draw the figure before comparing the final map and colorbar positions
        ax, colorbar_ax = vector.plot(column="value", ax="new", return_axes=True)
        assert colorbar_ax is not None
        ax.figure.canvas.draw()
        map_position = ax.get_position()
        colorbar_position = colorbar_ax.get_position()
        gap = colorbar_position.x0 - map_position.x1

        assert 0 <= gap < map_position.width / 2
        assert colorbar_position.height == pytest.approx(map_position.height)
        plt.close(ax.figure)

    def test_plot__reference_panel_size(self) -> None:
        """Checks that a reference-matched vector plot has the same map and colorbar size as a raster plot."""

        # Create a wide reference that makes equal-aspect axes shorter than their original subplot slots
        reference = gu.Raster.from_array(np.ones((10, 100)), Affine(1, 0, 0, 0, -1, 10), 3857)
        dataframe = gpd.GeoDataFrame(
            {"value": [0, 1]},
            geometry=gpd.points_from_xy([0, 100], [0, 1]),
            crs=3857,
        )
        vector = gu.Vector(dataframe)

        # Plot the same extent in paired panels whose titles occupy one and two lines
        fig, axes = plt.subplots(1, 2)
        axes[0].set_title("Raster")
        raster_ax, raster_colorbar_ax = reference.plot(ax=axes[0], max_pixels=None, return_axes=True)
        axes[1].set_title("Polygonized\nvector")
        vector_ax, vector_colorbar_ax = vector.plot(
            ref=reference,
            ax=axes[1],
            column="value",
            return_axes=True,
        )
        assert raster_colorbar_ax is not None
        assert vector_colorbar_ax is not None
        fig.tight_layout()
        fig.canvas.draw()

        # Matching data bounds and non-resizing colorbars should give both panels the same size
        raster_position = raster_ax.get_position()
        vector_position = vector_ax.get_position()
        assert vector_position.y0 == pytest.approx(raster_position.y0)
        assert vector_position.width == pytest.approx(raster_position.width)
        assert vector_position.height == pytest.approx(raster_position.height)
        assert vector_colorbar_ax.get_position().width == pytest.approx(raster_colorbar_ax.get_position().width)
        assert vector_colorbar_ax.get_position().height == pytest.approx(raster_colorbar_ax.get_position().height)

        plt.close(fig)

    def test_plot__projected_tick_spacing(self) -> None:
        """Checks that long projected coordinates do not overlap on a narrow plot."""

        # Create one narrow panel with six-digit easting labels like those in documentation figures
        dataframe = gpd.GeoDataFrame(
            {"value": [0, 1]},
            geometry=gpd.points_from_xy([470_000, 510_000], [3_080_000, 3_120_000]),
            crs=32632,
        )
        vector = gu.Vector(dataframe)
        fig, axes = plt.subplots(1, 3, figsize=(4, 3))

        # Plot in the first panel and compare the rendered bounds of neighboring labels
        vector.plot(ax=axes[0], column="value")
        fig.canvas.draw()
        labels = [label for label in axes[0].get_xticklabels() if label.get_visible() and label.get_text()]
        label_boxes = [label.get_window_extent() for label in labels]

        assert all(not first.overlaps(second) for first, second in zip(label_boxes[:-1], label_boxes[1:]))
        plt.close(fig)

    def test_plot__explicit_colorbar_axes_title(self) -> None:
        """Checks that plot() uses supplied colorbar axes and title."""

        # Synthetic point data
        dataframe = gpd.GeoDataFrame({"value": [10, 20]}, geometry=gpd.points_from_xy([0, 1], [0, 1]), crs=32632)
        vector = gu.Vector(dataframe)

        # Axes to supply to the function
        figure, (map_axes, colorbar_axes) = plt.subplots(1, 2)

        # We plot and check the axes passed are used, as well as the colorbar
        try:
            returned_map, returned_colorbar = vector.plot(
                column="value", ax=map_axes, cax=colorbar_axes, cbar_title="Value", return_axes=True
            )

            assert returned_map is map_axes
            assert returned_colorbar is colorbar_axes
            assert colorbar_axes.get_ylabel() == "Value"
        finally:
            plt.close(figure)

    def test_plot__match_reference_extent(self) -> None:
        """Checks that a reference object keeps its full extent after a larger vector overlay."""

        # Plot a raster before a vector whose point extent is much larger
        reference = gu.Raster.from_array(np.ones((10, 10)), Affine(1, 0, 0, 0, -1, 10), 4326)
        dataframe = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy([-100, 100], [-100, 100]),
            crs=4326,
        )
        vector = gu.Vector(dataframe)
        fig, ax = plt.subplots()
        reference.plot(ax=ax, max_pixels=None, add_cbar=False)
        vector.plot(ref=reference, ax=ax, add_cbar=False)

        # The reference controls both the common CRS and the final visible bounds
        assert ax.get_xlim() == pytest.approx((reference.bounds.left, reference.bounds.right))
        assert ax.get_ylim() == pytest.approx((reference.bounds.bottom, reference.bounds.top))
        plt.close(fig)

    def test_plot__crs_reference(self) -> None:
        """Checks that a CRS reference reprojects the plot without imposing reference bounds."""

        # Create one geographic point whose projected coordinate is clearly distinct from one degree
        dataframe = gpd.GeoDataFrame(geometry=gpd.points_from_xy([1], [1]), crs=4326)
        vector = gu.Vector(dataframe)

        # Pass a CRS string through the same ref argument accepted by raster and point cloud plots
        ax, _ = vector.plot(ref="EPSG:3857", add_cbar=False, ax="new", return_axes=True)
        offsets = np.asarray(ax.collections[0].get_offsets())

        assert offsets[0, 0] > 100_000
        assert vector.crs.to_epsg() == 4326
        plt.close(ax.figure)

    def test_plot__deprecated_ref_crs(self) -> None:
        """Checks that the old ref_crs argument warns and uses only the reference CRS."""

        dataframe = gpd.GeoDataFrame(geometry=gpd.points_from_xy([1], [1]), crs=4326)
        vector = gu.Vector(dataframe)
        reference = gu.Raster.from_array(np.ones((2, 2)), Affine(1, 0, 0, 0, -1, 2), 3857)

        # Check that ref_crs changes the CRS without using the reference bounds
        with pytest.warns(DeprecationWarning, match="Argument 'ref_crs' is deprecated"):
            ax, _ = vector.plot(ref_crs=reference, add_cbar=False, ax="new", return_axes=True)
        offsets = np.asarray(ax.collections[0].get_offsets())

        assert offsets[0, 0] > 100_000
        assert ax.get_xlim()[0] > reference.bounds.right
        plt.close(ax.figure)

        # Reject calls that pass both the old and new arguments
        with pytest.raises(TypeError, match="received both 'ref' and deprecated 'ref_crs'"):
            vector.plot(ref=3857, ref_crs=reference, add_cbar=False)

    @pytest.mark.parametrize("axis", [0, "existing"])
    def test_plot__error_invalid_axes(self, axis: Any) -> None:
        """Checks an error is raised for invalid axes."""

        # Synthetic point cloud
        dataframe = gpd.GeoDataFrame(geometry=gpd.points_from_xy([0], [0]), crs=32632)
        vector = gu.Vector(dataframe)

        # Check for error with wrong axis
        with pytest.raises(ValueError, match="ax must be a matplotlib.axes.Axes instance"):
            vector.plot(ax=axis, add_cbar=False)
