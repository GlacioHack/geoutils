"""Tests on Xarray accessor mirroring Raster API."""

import warnings

import rioxarray as rioxr

from geoutils import examples, Raster, open_raster

class TestAccessor:


    def test_open_raster(self):




class TestConsistencyRasterAccessor:

    # Test over many different rasters
    landsat_b4_path = examples.get_path("everest_landsat_b4")



    @pytest.mark.parametrize("path_raster", [landsat_b4_path])  # type: ignore
    @pytest.mark.parametrize("method", nongeo_properties)  # type: ignore
    def test_properties(self, path_raster: str, method: str) -> None:
        """Check non-geometric properties are consistent with GeoPandas."""

        # Open
        ds = open_raster(path_raster)
        raster = Raster(path_raster)

        # Remove warnings about operations in a non-projected system, and future changes
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)

        # Get method for each class
        output_raster = getattr(raster, method)
        output_ds = getattr(ds, method)

        # Assert equality
