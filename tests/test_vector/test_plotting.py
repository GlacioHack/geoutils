"""Test vector plotting layout and reference matching."""

from __future__ import annotations

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
        ax, colorbar_ax = vector.plot(column="value", return_axes=True)
        assert colorbar_ax is not None
        ax.figure.canvas.draw()
        map_position = ax.get_position()
        colorbar_position = colorbar_ax.get_position()
        gap = colorbar_position.x0 - map_position.x1

        assert 0 <= gap < map_position.width / 2
        assert colorbar_position.height == pytest.approx(map_position.height)
        plt.close(ax.figure)

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
        vector.plot(ref="EPSG:3857", add_cbar=False)
        offsets = np.asarray(plt.gca().collections[0].get_offsets())

        assert offsets[0, 0] > 100_000
        assert vector.crs.to_epsg() == 4326
        plt.close()
