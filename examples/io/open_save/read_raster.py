"""
Open/save a raster
==================

This example demonstrates the instantiation of a raster through :class:`~geoutils.Raster` and saving with :func:`~geoutils.Raster.to_file`.
"""

# %%
# We open an example raster. The data is, by default, unloaded.
import geoutils as gu

filename_rast = gu.examples.get_path("everest_landsat_b4")
rast = gu.Raster(filename_rast)
rast

# %%
# A raster is composed of four main attributes: a :class:`~geoutils.Raster.data` array, an affine :class:`~geoutils.Raster.transform`,
# a coordinate reference system :class:`~geoutils.Raster.crs` and a :class:`~geoutils.Raster.nodata` value.
# All other attributes are derivatives of those or the file on disk, and can be found in the :ref:`dedicated section of the API<api-raster-attrs>`. See also :ref:`raster-class`.

# %%
#
# .. note::
#        A raster can also be instantiated with a :class:`rasterio.io.DatasetReader` or a :class:`rasterio.io.MemoryFile`, see :ref:`sphx_glr_io_examples_import_export_import_raster.py`.
#
# We can print more info on the raster.
rast.info()

# %%
# The data will be loaded explicitly by any function requiring its :attr:`~geoutils.Raster.data`, such as :func:`~geoutils.Raster.show`.
rast.plot(cmap="Greys_r")

# %%
# Opening can be performed with several parameters, for instance choosing a single band with ``index`` and re-sampling with ``downsample``, to subset a 3-band
# raster to its second band, and using 1 pixel out of 4.
rast = gu.Raster(gu.examples.get_path("everest_landsat_rgb"), bands=2, downsample=4)
rast

# %%
# The data is not loaded by default, even if when specifying a band or re-sampling.
# We can load it explicitly by calling :func:`~geoutils.Raster.load` (could have also passed ``load_data=True`` to :class:`~geoutils.Raster`).
rast.load()
rast

# %%
# Finally, a raster is saved using :func:`~geoutils.Raster.to_file`:
rast.to_file("myraster.tif")
