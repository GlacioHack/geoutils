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
