"""Test raster construction from arrays."""

from typing import Literal

import numpy as np
import pytest
import xarray as xr
from affine import Affine

import geoutils as gu


class TestRasterConstructors:
    """Test module for raster construction with from_array()."""

    def test_from_array__default(self) -> None:
        """Checks that from_array() returns a DataArray by default, with the right values and georeferencing."""

        # Create from synthetic data
        values = np.arange(6).reshape(2, 3)
        transform = Affine(10, 0, 500000, 0, -10, 8600000)

        # Construct raster
        raster = gu.raster.from_array(values, transform, 32631)

        # Check default name, dimensions, values and georeferencing
        assert isinstance(raster, xr.DataArray)
        assert raster.name is None and raster.dims == ("y", "x")
        np.testing.assert_array_equal(raster.data, values)
        assert raster.rst.transform == transform and raster.rst.crs.to_epsg() == 32631
        assert raster.rst.is_loaded

    @pytest.mark.parametrize("as_object_type", ["dataarray", "dataset"])
    @pytest.mark.parametrize("data_name", [None, "elevation"])
    @pytest.mark.parametrize("bands", [1, 2])
    def test_from_array__bands_masks_and_metadata(
        self, as_object_type: Literal["dataarray", "dataset"], data_name: str | None, bands: int
    ) -> None:
        """Checks that every object type preserves bands, masked values and metadata."""

        # We create a masked integer raster with a rotated transform
        values = np.arange(bands * 6, dtype=np.uint16).reshape(bands, 2, 3)
        masked = np.ma.array(values, mask=values == 2)
        transform = Affine(10, 1, 500000, 2, -10, 8600000)
        tags = {"sensor": "example"}

        # Construct raster
        result = gu.raster.from_array(
            masked,
            tuple(transform)[:6],
            32631,
            nodata=-9999,
            area_or_point="Point",
            tags=tags,
            data_name=data_name,
            as_object_type=as_object_type,
        )
        raster = result[data_name or "raster"] if as_object_type == "dataset" else result

        # Check dimensions, masked values and metadata
        expected = masked.astype(float).filled(np.nan)
        expected = expected[0] if bands == 1 else expected
        np.testing.assert_array_equal(raster.data, expected)
        assert raster.dims == (("y", "x") if bands == 1 else ("band", "y", "x"))
        assert raster.rst.transform == transform and raster.rst.crs.to_epsg() == 32631
        assert raster.rst.nodata == -9999 and raster.rst.area_or_point == "Point"

        # Check rotated coordinates and unchanged input tags
        rows, columns = np.indices((2, 3))
        expected_x = transform.a * (columns + 0.5) + transform.b * (rows + 0.5) + transform.c
        expected_y = transform.d * (columns + 0.5) + transform.e * (rows + 0.5) + transform.f
        np.testing.assert_array_equal(raster.xc, expected_x)
        np.testing.assert_array_equal(raster.yc, expected_y)
        assert raster.attrs["sensor"] == "example" and tags == {"sensor": "example"}
        expected_name = data_name or "raster" if as_object_type == "dataset" else data_name
        assert raster.name == expected_name


class TestRasterConstructorsChunked:
    """Test module for lazy construction with from_array()."""

    @pytest.mark.parametrize("as_object_type", ["dataarray", "dataset"])
    @pytest.mark.parametrize("bands", [1, 2])
    def test_from_array__loading_laziness(
        self, as_object_type: Literal["dataarray", "dataset"], bands: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checks that lazy construction runs no tasks and exactly matches eager construction."""
        pytest.importorskip("dask")
        import dask.array as da
        from dask.callbacks import Callback

        # We use 5 x 7 pixels to check uneven chunks of 3/2 rows and 4/3 columns
        values = np.arange(bands * 35, dtype=float).reshape(bands, 5, 7)
        values[0, 1, 2] = np.nan
        chunks = da.from_array(values, chunks=(1, 3, 4))
        transform = Affine(10, 0, 500000, 0, -10, 8600000)
        assert isinstance(chunks, da.Array)

        # We construct both outputs, checking there is no task execution during lazy construction
        eager = gu.raster.from_array(values, transform, 32631, data_name="dem", as_object_type=as_object_type)
        monkeypatch.setitem(gu.config, "raster_return_type", as_object_type)
        with Callback(pretask=lambda *args: pytest.fail("Constructing rasters must not compute pixels")):
            lazy = gu.raster.from_array(chunks, transform, 32631, data_name="dem")
        raster = lazy.dem if as_object_type == "dataset" else lazy

        # Check loading, then equality
        assert not raster.rst.is_loaded and raster.chunks
        actual = lazy.compute()
        xr.testing.assert_identical(actual, eager)
        assert (actual.dem if as_object_type == "dataset" else actual).rst.is_loaded

        # Check computing output leaves the input graphs and lazy result unchanged
        assert not raster.rst.is_loaded and raster.chunks
        assert isinstance(chunks, da.Array)


class TestRasterConstructorsErrors:
    """Test module for errors in from_array()."""

    def test_from_array__error_invalid_type(self) -> None:
        """Checks an error is raised for an unsupported raster object type."""

        # Create from synthetic data
        values = np.ones((2, 3))
        transform = Affine(10, 0, 0, 0, -10, 20)
        with pytest.raises(ValueError, match="as_object_type"):
            gu.raster.from_array(values, transform, 32631, as_object_type="geodataframe")  # type: ignore[arg-type]
