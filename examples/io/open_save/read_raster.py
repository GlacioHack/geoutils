"""
Open/save a raster
==================

This example demonstrates opening a raster with :func:`~geoutils.open_raster` and saving it with the :class:`rst <geoutils.DataArrayRasterAccessor>` accessor.
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
# Downsampling uses nearest-neighbor reads and suitable stored overviews. A factor preserves exact pixel spacing,
# trimming incomplete intervals at the edges. A number is shorthand for the factor option:
preview = gu.open_raster(filename_rast, downsample={"factor": 4})
preview

# %%
# The same argument accepts a pixel size in CRS units, or an exact shape in (height, width) order.
# Shape requests always preserve the full extent. All these options leave the pixels unloaded until needed.
preview = gu.open_raster(filename_rast, downsample={"res": 120})
preview = gu.open_raster(filename_rast, downsample={"shape": (64, 128)})
preview

# %%
# Single-band rasters normally have dimensions (y, x). Set ``squeeze=False`` to preserve (band, y, x),
# including through raster operations:
with_band = gu.open_raster(filename_rast, squeeze=False)
with_band.rst.icrop((0, 0, 10, 10))

# %%
# By default, missing values become NaNs and integer values are converted to floating point. ``masked=False``
# reads stored values and their dtype, which is useful for rearranging bands and writing them back to disk.
# It emits a warning because GeoUtils operations that expect NaNs may return incorrect results with numeric nodata.
# An explicit ``mask_and_scale=True`` still decodes values through Rioxarray.
stored = gu.open_raster(filename_rast, masked=False)
stored

# %%
# The data is not loaded by default, even if when specifying a band or re-sampling.
# We can load it explicitly by calling :meth:`~geoutils.raster.base.RasterBase.load`.
rast.rst.load()
rast

# %%
# Finally, a raster is saved using :meth:`~geoutils.DataArrayRasterAccessor.to_file`:
rast.rst.to_file("myraster.tif")
