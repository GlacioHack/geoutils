"""
From/To Rasterio
================

This example demonstrates importing and exporting a :class:`rasterio.io.DatasetReader` or :class:`rasterio.io.DatasetReader` from and to a
:class:`~geoutils.Raster`.
"""

# %%
# A raster can be imported from a :class:`rasterio.io.DatasetReader` or :class:`rasterio.io.MemoryFile` simply by instantiating :class:`~geoutils.Raster`.
import geoutils as gu
import rasterio as rio

ds = rio.DatasetReader(gu.examples.get_path("everest_landsat_b4"))
rast = gu.Raster(ds)
rast

# %%
# The data is unloaded, as when instantiated with a filename.
# The data will be loaded explicitly by any function requiring its :attr:`~geoutils.Raster.data`, such as :func:`~geoutils.Raster.show`.
rast.show(cmap="Greys_r")

# %%
# We can also pass a :class:`rasterio.io.MemoryFile` during instantiation.

mem = rio.MemoryFile(open(gu.examples.get_path("everest_landsat_b4"), "rb"))
rast = gu.Raster(mem)
rast

# %%
# The data is, as expected, already in memory.
#
# Finally, we can export a :class:`~geoutils.Raster` to a :class:`rasterio.io.DatasetReader` of a :class:`rasterio.io.MemoryFile` using
# :class:`~geoutils.Raster.to_rio_dataset`

rast.to_rio_dataset()