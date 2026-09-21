"""Test distance functions at the interface of raster and vectors."""

from __future__ import annotations

import os
import tempfile
import warnings
from importlib.util import find_spec
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio as rio
from shapely.geometry import Polygon

import geoutils as gu
from geoutils._typing import NDArrayNum
from geoutils.multiproc import MultiprocConfig


def run_gdal_proximity(
    input_raster: gu.Raster, target_values: list[float] | None, distunits: str = "GEO"
) -> NDArrayNum:
    """Run GDAL's ComputeProximity and return the read numpy array."""
    # Rasterio strongly recommends against importing gdal along rio, so this is done here instead.
    from osgeo import gdal, gdalconst

    gdal.UseExceptions()

    # Initiate empty GDAL raster for proximity output
    drv = gdal.GetDriverByName("MEM")
    proxy_ds = drv.Create("", input_raster.shape[1], input_raster.shape[0], 1, gdal.GetDataTypeByName("Float32"))
    proxy_ds.GetRasterBand(1).SetNoDataValue(-9999)

    # Save input in temporary file to read with GDAL
    # (avoids the nightmare of setting nodata, transform, crs in GDAL format...)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, "input.tif")
        input_raster.to_file(temp_path)
        ds_raster_in = gdal.Open(temp_path, gdalconst.GA_ReadOnly)

        # Define GDAL options
        proximity_options = ["DISTUNITS=" + distunits]
        if target_values is not None:
            proximity_options.insert(0, "VALUES=" + ",".join([str(tgt) for tgt in target_values]))

        # Compute proximity
        gdal.ComputeProximity(ds_raster_in.GetRasterBand(1), proxy_ds.GetRasterBand(1), proximity_options)
        # Save array
        proxy_array = proxy_ds.GetRasterBand(1).ReadAsArray().astype("float32")
        proxy_array[proxy_array == -9999] = np.nan

        # Close GDAL datasets
        proxy_ds = None
        ds_raster_in = None

    return proxy_array


class TestDistance:
    """
    Test module for proximity().

    Chunked execution with Dask/MP is tested further below in TestDistanceChunked.
    """

    landsat_b4_path = gu.examples.get_path_test("everest_landsat_b4")
    landsat_b4_crop_path = gu.examples.get_path_test("everest_landsat_b4_cropped")
    everest_outlines_path = gu.examples.get_path_test("everest_rgi_outlines")
    aster_dem_path = gu.examples.get_path_test("exploradores_aster_dem")

    def test_proximity_vector(self) -> None:
        """
        The core functionality is already tested against GDAL in test_raster: just verify the vector-specific behaviour.
        #TODO: add an artificial test as well (mirroring TODO in test_raster)
        """

        vector = gu.Vector(self.everest_outlines_path)

        # -- Test 1: with a Raster provided --
        raster1 = gu.Raster(self.landsat_b4_crop_path)
        prox1 = vector.proximity(raster=raster1)

        # The proximity should have the same extent, resolution and CRS
        assert raster1.georeferenced_grid_equal(prox1)

        # Apply a geometry operation before calculating proximity to the derived geometry
        vector.boundary.proximity(raster=raster1)

        # -- Test 2: with no Raster provided, just grid size --

        # Default grid size
        vector.proximity()

        # With specific grid size
        vector.proximity(size=(100, 100))

        # Test all options, with both an artificial Raster (that has all target values) and a real Raster

    @pytest.mark.parametrize("distunits", ["GEO", "PIXEL"])
    # 0 and 1,2,3 are especially useful for the artificial Raster, and 112 for the real Raster
    @pytest.mark.parametrize("target_values", [[1, 2, 3], [0], [112], None])
    @pytest.mark.parametrize(
        "raster",
        [
            gu.Raster(landsat_b4_path),
            gu.Raster.from_array(
                np.arange(25, dtype="int32").reshape(5, 5), transform=rio.transform.from_origin(0, 5, 1, 1), crs=4326
            ),
        ],
    )
    def test_proximity_raster_against_gdal(
        self, distunits: str, target_values: list[float] | None, raster: gu.Raster
    ) -> None:
        """Test that proximity matches the results of GDAL for any parameter."""

        # TODO: When adding new rasters for tests, specify warning only for Landsat
        warnings.filterwarnings("ignore", message="Setting default nodata -99999 to mask non-finite values *")

        # We generate proximity with GDAL and GeoUtils
        gdal_proximity = run_gdal_proximity(raster, target_values=target_values, distunits=distunits)
        # We translate distunits GDAL option into its GeoUtils equivalent
        if distunits == "GEO":
            distance_unit = "georeferenced"
        else:
            distance_unit = "pixel"
        geoutils_proximity = (
            raster.proximity(distance_unit=distance_unit, target_values=target_values)
            .data.data.squeeze()
            .astype("float32")
        )

        # The results should be the same in all cases
        try:
            # In some cases, the proximity differs slightly (generally <1%) for complex settings
            # (Landsat Raster with target of 112)
            # It looks like GDAL might not have the right value,
            # so this particular case is treated differently in tests
            if target_values is not None and target_values[0] == 112 and raster.name is not None:
                # Get index and number of not almost equal point (tolerance of 10-4)
                ind_not_almost_equal = np.abs(gdal_proximity - geoutils_proximity) > 1e-04
                nb_not_almost_equal = np.count_nonzero(ind_not_almost_equal)
                # Check that this is a minority of points (less than 0.5%)
                assert nb_not_almost_equal < 0.005 * raster.width * raster.height

                # Replace these exceptions by zero in both
                gdal_proximity[ind_not_almost_equal] = 0.0
                geoutils_proximity[ind_not_almost_equal] = 0.0
                # Check that all the rest is almost equal
                assert np.allclose(gdal_proximity, geoutils_proximity, atol=1e-04, equal_nan=True)

            # Otherwise, results are exactly equal
            else:
                assert np.array_equal(gdal_proximity, geoutils_proximity, equal_nan=True)

        # For debugging
        except Exception as exception:
            import matplotlib.pyplot as plt

            # Plotting the xdem and GDAL attributes for comparison (plotting "diff" can also help debug)
            plt.subplot(121)
            plt.imshow(gdal_proximity)
            # plt.imshow(np.abs(gdal_proximity - geoutils_proximity)>0.1)
            plt.colorbar()
            plt.subplot(122)
            plt.imshow(geoutils_proximity)
            # plt.imshow(raster.data.data == 112)
            plt.colorbar()
            plt.show()

            # ind_not_equal = np.abs(gdal_proximity - geoutils_proximity)>0.1
            # print(gdal_proximity[ind_not_equal])
            # print(geoutils_proximity[ind_not_equal])

            raise exception

    def test_proximity_raster_parameters(self) -> None:
        """
        Test that new (different to GDAL's) proximity parameters run.
        No need to test the results specifically, as those rely entirely on the previous test with GDAL,
        and tests in rasterize and shapely.
        #TODO: Maybe add one test with an artificial vector to check it works as intended
        """

        # -- Test 1: with self's Raster alone --
        raster1 = gu.Raster(self.landsat_b4_path)
        prox1 = raster1.proximity()

        # The raster should have the same extent, resolution and CRS
        assert raster1.georeferenced_grid_equal(prox1)

        # It should change with target values specified
        prox2 = raster1.proximity(target_values=[255])
        assert not np.array_equal(prox1.data, prox2.data)

        # -- Test 2: with a vector provided --
        vector = gu.Vector(self.everest_outlines_path)

        # Use the vector's current geometry as distance targets
        raster1.proximity(vector=vector)

        # Apply a geometry operation before calculating proximity to the derived geometry
        raster1.proximity(vector=vector.boundary)

        # Paths to example data

    # Mask without nodata
    mask_landsat_b4 = gu.Raster(landsat_b4_path) > 125
    # Mask with nodata
    mask_aster_dem = gu.Raster(aster_dem_path) > 2000
    # Mask from an outline
    mask_everest = gu.Vector(everest_outlines_path).create_mask(gu.Raster(landsat_b4_path))

    @pytest.mark.parametrize("mask", [mask_landsat_b4, mask_aster_dem, mask_everest])
    def test_proximity_mask(self, mask: gu.Raster) -> None:
        mask_orig = mask.copy()
        # Run default
        rast = mask.proximity()
        # Check the dtype of the original mask was properly reconverted
        assert mask.data.dtype == bool
        # Check the original mask was not modified during reprojection
        assert mask_orig.raster_equal(mask)

        # Check that output is cast back into a raster
        assert isinstance(rast, gu.Raster)

    def test_proximity__max_distance(self) -> None:
        """Checks that max_distance sets nodata for pixels values beyond it."""

        # Create one target pixel in a small raster (every expected distance can be derived from its row/column)
        values = np.zeros((5, 7), dtype=np.uint8)
        values[2, 3] = 1
        transform = rio.transform.from_origin(0, 5, 1, 1)
        raster = gu.Raster.from_array(values, transform=transform, crs=32610)

        # Calculate distances no farther than two pixels from the target
        result = raster.proximity(target_values=[1], distance_unit="pixel", max_distance=2)

        # Compare with the distances computed manually
        rows, columns = np.indices(values.shape)
        distances = np.hypot(rows - 2, columns - 3)
        expected = np.where(distances <= 2, distances, np.nan)
        np.testing.assert_allclose(result.to_nanarray(), expected, equal_nan=True)

    @pytest.mark.parametrize("max_distance", [-1, np.inf, np.nan])
    def test_proximity__error_invalid_max_distance(self, max_distance: float) -> None:
        """Checks that proximity raises an error for negative and non-finite maximum distances."""

        # Create the smallest raster containing both a target and a non-target cell
        values = np.array([[1, 0]], dtype=np.uint8)
        raster = gu.Raster.from_array(values, transform=rio.transform.from_origin(0, 1, 1, 1), crs=32610)

        # Reject a distance that cannot define a finite, non-negative overlap
        with pytest.raises(ValueError, match="max_distance must be non-negative and finite"):
            raster.proximity(distance_unit="pixel", max_distance=max_distance)


@pytest.mark.skipif(find_spec("dask") is None, reason="Only runs if dask is installed.")
class TestDistanceChunked:
    """Test module for proximity values and loading behavior with Dask and multiprocessing."""

    @pytest.mark.parametrize(
        "distance_unit,max_distance",
        [("pixel", 3.0), ("georeferenced", 7.0)],
    )
    def test_proximity__chunked_raster_backends_equal(
        self,
        distance_unit: str,
        max_distance: float,
        tmp_path: Path,
    ) -> None:
        """Checks that chunked raster proximity stays lazy and exactly matches the eager result."""

        import dask.array as da

        # Write targets to file
        values = np.zeros((9, 11), dtype=np.uint8)
        values[3, 5] = 1
        values[7, 9] = 1
        transform = rio.transform.from_origin(0, 27, 2, 3)
        source_path = tmp_path / "proximity_source.tif"
        source = gu.Raster.from_array(values, transform=transform, crs=32610)
        source.to_file(source_path)

        # Calculate the same proximity distances eagerly, lazily with Dask, and with multiprocessing
        options = {"target_values": [1], "distance_unit": distance_unit, "max_distance": max_distance}
        expected = source.proximity(**options)
        dask_source = gu.open_raster(str(source_path), chunks={"y": 4, "x": 5})
        multiproc_source = gu.Raster(source_path)
        dask_result = dask_source.rst.proximity(**options)
        mp_config = MultiprocConfig(chunks=(4, 5), outfile=str(tmp_path / "proximity_multiproc.tif"))
        multiproc_result = multiproc_source.proximity(**options, mp_config=mp_config)

        # Check that the chunked inputs and outputs remain lazy before reading their values
        assert isinstance(dask_result.data, da.Array)
        assert not dask_source._in_memory
        assert not multiproc_source.is_loaded
        assert not multiproc_result.is_loaded

        # Require the same distances and missing cells from each backend
        expected_values = expected.to_nanarray()
        np.testing.assert_array_equal(dask_result.compute().data, expected_values)
        np.testing.assert_array_equal(multiproc_result.to_nanarray(), expected_values)
        assert multiproc_result.transform == expected.transform
        assert multiproc_result.crs == expected.crs

        # Reading either result must leave its source unloaded
        assert not dask_source._in_memory
        assert not multiproc_source.is_loaded

    def test_proximity__chunked_vector_backends_equal(self, tmp_path: Path) -> None:
        """Checks that chunked proximity to a derived vector boundary exactly matches eager execution."""

        import dask.array as da

        # Define a polygon with boundary crossing several chunks
        values = np.zeros((9, 11), dtype=np.uint8)
        transform = rio.transform.from_origin(0, 9, 1, 1)
        source_path = tmp_path / "vector_proximity_source.tif"
        source = gu.Raster.from_array(values, transform=transform, crs=32610)
        source.to_file(source_path)
        geometry = Polygon([(2, 2), (9, 2), (9, 7), (2, 7)])
        vector = gu.Vector(gpd.GeoDataFrame(geometry=[geometry], crs=32610))
        boundary = vector.boundary

        # Calculate distances from the explicitly derived polygon boundary through every backend
        options = {"distance_unit": "pixel", "max_distance": 3.0}
        expected = boundary.proximity(raster=source, **options)
        dask_source = gu.open_raster(str(source_path), chunks={"y": 4, "x": 5})
        multiproc_source = gu.Raster(source_path)
        dask_result = boundary.proximity(raster=dask_source.rst, **options)
        mp_config = MultiprocConfig(chunks=(4, 5), outfile=str(tmp_path / "vector_proximity_multiproc.tif"))
        multiproc_result = boundary.proximity(raster=multiproc_source, **options, mp_config=mp_config)

        # Check lazy output and compare both chunked calculations with the eager vector result
        assert isinstance(dask_result.data, da.Array)
        assert not multiproc_source.is_loaded
        assert not multiproc_result.is_loaded
        expected_values = expected.to_nanarray()
        np.testing.assert_array_equal(dask_result.compute().data, expected_values)
        np.testing.assert_array_equal(multiproc_result.to_nanarray(), expected_values)
        assert not dask_source._in_memory
        assert not multiproc_source.is_loaded

    def test_proximity__error_chunked_backend_configuration(self, tmp_path: Path) -> None:
        """Checks errors raised by chunked proximity."""

        # Open one file with Dask chunks so proximity must use a finite overlap
        values = np.zeros((5, 6), dtype=np.uint8)
        values[2, 3] = 1
        source_path = tmp_path / "proximity_configuration.tif"
        gu.Raster.from_array(values, rio.transform.from_origin(0, 5, 1, 1), 32610).to_file(source_path)
        dask_source = gu.open_raster(str(source_path), chunks={"y": 2, "x": 3})

        # Error if overlap not defined for chunked
        with pytest.raises(ValueError, match="max_distance must be provided"):
            dask_source.rst.proximity(distance_unit="pixel")
        # Error if MP/Dask used at the same time
        mp_config = MultiprocConfig(chunks=(2, 3), outfile=str(tmp_path / "unused.tif"))
        with pytest.raises(ValueError, match="Cannot use Multiprocessing and Dask simultaneously"):
            dask_source.rst.proximity(distance_unit="pixel", max_distance=2, mp_config=mp_config)
