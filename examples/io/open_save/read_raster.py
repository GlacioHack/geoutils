"""
Open/save a raster
==================

This example demonstrates opening a raster with :func:`~geoutils.open_raster` and saving it with the :class:`rst <geoutils.RasterAccessor>` accessor.
"""

# %%
# We open an example raster. The data is, by default, unloaded.
import geoutils as gu

filename_rast = gu.examples.get_path("everest_landsat_b4")
rast = gu.open_raster(filename_rast)
rast

# %%
# A raster is composed of four main attributes: a data array, an affine transform, a coordinate reference system and
# a nodata value.
# All other attributes are derivatives of those or the file on disk, and can be found in the :ref:`dedicated section of the API<api-raster-attrs>`. See also :ref:`raster-class`.

# %%
#
# .. note::
#        A raster can also be opened from a :class:`rasterio.io.DatasetReader` or a :class:`rasterio.io.MemoryFile`, see :ref:`sphx_glr_io_examples_import_export_import_raster.py`.
#
# We can print more info on the raster.
rast.rst.info()

# %%
# The data will be loaded explicitly by any function requiring its values, such as plotting.
rast.rst.plot(cmap="Greys_r")

# %%
# Opening can be performed with several parameters, for instance choosing a single band with ``index`` and re-sampling with ``downsample``, to subset a 3-band
# raster to its second band, and using 1 pixel out of 4.
rast = gu.open_raster(gu.examples.get_path("everest_landsat_rgb"), downsample=4).sel(band=2)
rast

# %%
# The data is not loaded by default, even if when specifying a band or re-sampling.
# We can load it explicitly by calling :meth:`~geoutils.raster.base.RasterBase.load`.
rast.rst.load()
rast

# %%
# Finally, a raster is saved using :meth:`~geoutils.RasterAccessor.to_file`:
rast.rst.to_file("myraster.tif")
