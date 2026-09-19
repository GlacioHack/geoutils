"""Test raster plotting."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from importlib.util import find_spec

import matplotlib.pyplot as plt
import numpy as np
import pytest
from affine import Affine

import geoutils as gu
from geoutils import examples


@pytest.fixture(autouse=True)
def close_matplotlib_figures() -> Iterator[None]:
    """Close every Matplotlib figure before and after each raster plotting test."""

    # Start from empty global plotting state even when another test module left a figure open
    plt.close("all")
    yield

    # Keep figures created by this test from affecting later tests
    plt.close("all")


class TestPlot:
    """
    Test module for raster plot appearance, display limits, CRS matching, and validation.

    Checks for Dask/MP are done further below in TestPlotChunked.
    """

    landsat_b4_path = examples.get_path_test("everest_landsat_b4")
    landsat_b4_crop_path = examples.get_path_test("everest_landsat_b4_cropped")
    landsat_rgb_path = examples.get_path_test("everest_landsat_rgb")
    aster_dem_path = examples.get_path_test("exploradores_aster_dem")

    @pytest.mark.parametrize("example", [landsat_b4_path, landsat_b4_crop_path, aster_dem_path])
    @pytest.mark.parametrize("figsize", np.arange(2, 20, 2))
    def test_plot__colorbar_height(self, example: str, figsize: int) -> None:
        """Checks that the colorbar height matches the raster axes height."""

        # Plot a raster with a colorbar at several figure sizes
        raster = gu.Raster(example)
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        raster.plot(ax=ax, add_cbar=True)
        fig.axes[0].set_axis_off()
        fig.axes[1].set_axis_off()

        # Height of the main plot and colorbar should be equal
        plot_height = fig.axes[0].get_tightbbox().height
        colorbar_height = fig.axes[1].get_tightbbox().height
        plt.close(fig)

        assert plot_height == pytest.approx(colorbar_height)

    def test_plot__native_single_band(self) -> None:
        """Checks that disabling the pixel limit plots every source value at the source extent."""

        # Open one band without loading it and request complete grid (max_pixels = None)
        raster = gu.Raster(self.landsat_b4_path)
        ax, _ = raster.plot(ax="new", max_pixels=None, return_axes=True)

        # Check the image values, orientation, and projected extent match that of the full raster
        image = ax.get_images()[0]
        assert np.array_equal(image.get_array(), np.flip(raster.get_nanarray(), axis=0), equal_nan=True)
        assert image.origin == "lower"
        assert image.get_extent() == [
            raster.bounds.left,
            raster.bounds.right,
            raster.bounds.bottom,
            raster.bounds.top,
        ]
        plt.close()

    def test_plot__native_rgb_and_band_selection(self) -> None:
        """Checks that RGB bands and one selected band keep their established image layouts."""

        # Plot every RGB band without reducing the source grid
        raster = gu.Raster(self.landsat_rgb_path)
        ax, colorbar_axes = raster.plot(ax="new", max_pixels=None, return_axes=True)
        image = ax.get_images()[0]

        # RGB values move the band dimension last and do not create a colorbar
        expected_rgb = np.flip(np.moveaxis(raster.get_nanarray(), 0, -1), axis=0)
        assert np.array_equal(image.get_array(), expected_rgb, equal_nan=True)
        assert colorbar_axes is None
        assert raster.data.shape[0] == 3
        plt.close()

        # Plot one band with the same one-based band selection as before
        ax = plt.subplot(111)
        raster.plot(bands=1, cmap="gray", ax=ax, add_cbar=False, title="Test", max_pixels=None)
        image = ax.get_images()[0]
        assert np.array_equal(image.get_array(), np.flip(raster.get_nanarray()[0], axis=0), equal_nan=True)
        plt.close()

    def test_plot__axes_limits_and_save(self) -> None:
        """Checks that plot options, returned axes, and direct figure saving work properly."""

        # Plot with user input for color limits and colorbar
        raster = gu.Raster(self.landsat_b4_path)
        ax = plt.subplot(111)
        returned_ax, colorbar_ax = raster.plot(
            cmap="gray",
            vmin=40,
            vmax=220,
            cbar_title="Custom cbar",
            ax=ax,
            return_axes=True,
        )
        assert returned_ax is ax
        assert colorbar_ax is not None
        plt.close()

        # Check save through the existing convenience argument
        with tempfile.TemporaryDirectory() as directory:
            filename = os.path.join(directory, "test.png")
            raster.plot(savefig_fname=filename)
            assert os.path.isfile(filename)
        plt.close()

    def test_plot__pixel_budget(self) -> None:
        """Checks that a max number of pixels reduces the temporary display grid."""

        # Create a rectangular raster large enough to require reduction
        values = np.arange(200 * 100, dtype=np.float32).reshape(100, 200)
        raster = gu.Raster.from_array(values, transform=Affine(2, 0, 0, 0, -2, 200), crs=32632)

        # Limit the rendered image while keeping the source values and grid unchanged
        raster.plot(max_pixels=2_000, resampling="nearest", add_cbar=False)
        plotted = plt.gca().get_images()[0].get_array()
        assert plotted.size <= 2_000
        assert plotted.shape[0] < raster.height
        assert raster.shape == (100, 200)
        assert np.array_equal(raster.data, values)
        plt.close()

    def test_plot__automatic_pixel_budget_and_interpolation_config(self) -> None:
        """Checks that automatic downsampling follows display size and Matplotlib's interpolation configuration."""

        # Create a raster much larger than the rendered axes and configure a global interpolation method
        values = np.arange(1_000 * 2_000, dtype=np.float32).reshape(1_000, 2_000)
        raster = gu.Raster.from_array(values, transform=Affine(1, 0, 0, 0, -1, 1_000), crs=32632)
        fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
        axes_bounds = ax.get_window_extent()
        axes_pixels = (axes_bounds.width, axes_bounds.height)

        # Plot with the automatic default so vector outputs embed only a display-sized raster
        with plt.rc_context({"image.interpolation": "nearest"}):
            raster.plot(ax=ax, add_cbar=False)
        image = ax.get_images()[0]

        assert image.get_array().shape[1] <= axes_pixels[0]
        assert image.get_array().shape[0] <= axes_pixels[1]
        assert image.get_interpolation() == "nearest"
        plt.close(fig)

    def test_plot__aspect(self) -> None:
        """Checks that a user input for aspect overrides the default "equal"."""

        # Plot a rectangular raster while allowing Matplotlib to fill the available axes
        raster = gu.Raster.from_array(np.ones((5, 10)), Affine(1, 0, 0, 0, -1, 5), 32632)
        raster.plot(aspect="auto", add_cbar=False)

        assert plt.gca().get_aspect() == "auto"
        plt.close()

    def test_plot__match_reference_crs(self) -> None:
        """Checks a match reference input is used properly for CRS and bounds."""

        # Create a geographic raster and a projected raster to use as the CRS reference
        source = gu.Raster.from_array(
            np.arange(100, dtype=np.float32).reshape(10, 10), Affine(0.1, 0, 0, 0, -0.1, 1), 4326
        )
        reference = gu.Raster.from_array(np.ones((2, 2)), Affine(1_000, 0, 0, 0, -1_000, 2_000), 3857)
        original_bounds = source.bounds

        # Reproject only the bounded display raster
        source.plot(ref=reference, max_pixels=100, add_cbar=False)
        ax = plt.gca()
        extent = ax.get_images()[0].get_extent()
        assert source.crs.to_epsg() == 4326
        assert source.bounds == original_bounds
        assert extent != [original_bounds.left, original_bounds.right, original_bounds.bottom, original_bounds.top]
        assert ax.get_xlim() == pytest.approx((reference.bounds.left, reference.bounds.right))
        assert ax.get_ylim() == pytest.approx((reference.bounds.bottom, reference.bounds.top))
        plt.close()

    def test_plot__accessor(self) -> None:
        """Checks that RasterAccessor exposes the shared plotting implementation."""

        # Build a DataArray raster and plot a reduced grid through .rst
        values = np.arange(100, dtype=np.float32).reshape(10, 10)
        data_array = gu.RasterAccessor.from_array(values, Affine(1, 0, 0, 0, -1, 10), 32632)
        data_array.rst.plot(max_pixels=25, add_cbar=False)

        assert plt.gca().get_images()[0].get_array().shape == (5, 5)
        plt.close()

    def test_plot__accessor_after_isel(self) -> None:
        """Checks that plotting an indexed DataArray uses its current values and extent."""

        # Select an offset window so both the values and georeferenced bounds differ from the opened array
        values = np.arange(100, dtype=np.float32).reshape(10, 10)
        data_array = gu.RasterAccessor.from_array(values, Affine(2, 0, 100, 0, -2, 220), 32632)
        subset = data_array.isel(x=slice(2, 8), y=slice(1, 7))

        # Plot the complete current window rather than using metadata from the original array
        subset.rst.plot(max_pixels=None, add_cbar=False)
        image = plt.gca().get_images()[0]
        left, bottom, right, top = subset.rio.bounds()
        assert np.array_equal(image.get_array(), np.flip(subset.values, axis=0))
        assert image.get_extent() == [left, right, bottom, top]
        plt.close()

    def test_plot__without_crs(self) -> None:
        """Checks that a raster without a CRS keeps its native coordinate grid when plotted."""

        # Plot a small unreferenced raster using its affine coordinates
        raster = gu.Raster.from_array(np.arange(25).reshape(5, 5), Affine(2, 0, 10, 0, -2, 20), crs=None)
        raster.plot(max_pixels=None, add_cbar=False)

        assert plt.gca().get_images()[0].get_extent() == [10.0, 20.0, 10.0, 20.0]
        plt.close()

    def test_plot__errors(self) -> None:
        """Checks that we raise clear errors for invalid bands, axes, limits, and max pixels."""

        # Open single band and RGB rasters for band validation
        raster = gu.Raster(self.landsat_b4_path)
        rgb = gu.Raster(self.landsat_rgb_path)

        with pytest.raises(ValueError, match="Only single-band or 3/4-band"):
            rgb.plot(bands=(1, 2))
        with pytest.raises(ValueError, match="Index must be in range"):
            raster.plot(bands=2)
        with pytest.raises(ValueError, match="Index must be int, tuple or None"):
            raster.plot(bands="wrong_type")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="vmin or vmax cannot be converted to float"):
            raster.plot(vmin="wrong_type", vmax="wrong_type")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="ax must be a matplotlib.axes.Axes instance"):
            raster.plot(ax="wrong_type")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="max_pixels must be"):
            raster.plot(max_pixels=0)
        plt.close("all")

    @pytest.mark.skipif(find_spec("matplotlib") is not None, reason="Only runs if matplotlib is missing.")
    def test_plot__missing_dependency(self) -> None:
        """Checks that plotting without Matplotlib raises the optional dependency error."""

        raster = gu.Raster(self.landsat_b4_path)
        with pytest.raises(ImportError, match="Optional dependency 'matplotlib' required"):
            raster.plot()


@pytest.mark.skipif(find_spec("dask") is None, reason="Only runs if dask is installed.")
class TestPlotChunked:
    """Test module for lazy raster plotting: checks the out-of-memory computation, and equality with eager."""

    aster_dem_path = examples.get_path_test("exploradores_aster_dem")

    def test_plot__loading_laziness(self) -> None:
        """Checks that plotting computes a downsampled image without loading the source."""

        # Open the same raster with spatial chunks and as an eager Rioxarray backend
        lazy = gu.open_raster(self.aster_dem_path, chunks={"x": 100, "y": 100})
        eager = gu.open_raster(self.aster_dem_path)
        assert not lazy._in_memory

        # Plot both inputs with the same grid budget and nearest-neighbor resampling
        lazy.rst.plot(max_pixels=4_000, resampling="nearest", add_cbar=False)
        lazy_values = np.asarray(plt.gca().get_images()[0].get_array())
        plt.close()
        eager.rst.plot(max_pixels=4_000, resampling="nearest", add_cbar=False)
        eager_values = np.asarray(plt.gca().get_images()[0].get_array())

        # The source stays lazy and the computed display agrees exactly with eager reprojection
        assert not lazy._in_memory
        assert lazy_values.size <= 4_000
        assert lazy_values.shape == eager_values.shape
        np.testing.assert_array_equal(lazy_values, eager_values)
        plt.close()

    def test_plot__match_reference_loading_laziness(self) -> None:
        """Checks that performing CRS matching and resampling preserve input laziness."""

        # Open a source in chunks and retain its original georeferencing
        source = gu.open_raster(self.aster_dem_path, chunks={"x": 80, "y": 120})
        original_crs = source.rio.crs
        original_shape = source.rio.shape

        # Reproject only the display grid to a geographic CRS
        source.rst.plot(ref=4326, max_pixels=2_000, add_cbar=False)
        plotted = plt.gca().get_images()[0].get_array()

        assert not source._in_memory
        assert source.rio.crs == original_crs
        assert source.rio.shape == original_shape
        assert plotted.size <= 2_000
        plt.close()
