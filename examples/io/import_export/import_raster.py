"""
From/to Rasterio
================

This example demonstrates importing and exporting a :class:`rasterio.io.DatasetReader` or :class:`rasterio.io.DatasetReader` from and to a
:class:`~geoutils.Raster`.
"""

import rasterio as rio

# %%
# A raster can be imported from a :class:`rasterio.io.DatasetReader` or :class:`rasterio.io.MemoryFile` simply by instantiating :class:`~geoutils.Raster`.
import geoutils as gu

filename_rast = gu.examples.get_path("exploradores_aster_dem")
ds = rio.DatasetReader(filename_rast)
rast = gu.Raster(ds)
rast

# %%
# The data is unloaded, as when instantiated with a filename.
# The data will be loaded explicitly by any function requiring its :attr:`~geoutils.Raster.data`, such as :func:`~geoutils.Raster.show`.
rast.plot(cmap="terrain")

# %%
# We can also pass a :class:`rasterio.io.MemoryFile` during instantiation.

mem = rio.MemoryFile(open(filename_rast, "rb"))
rast = gu.Raster(mem)
rast

# %%
# The data is, as expected, already in memory.
#
# Finally, we can export a :class:`~geoutils.Raster` to a :class:`rasterio.io.DatasetReader` of a :class:`rasterio.io.MemoryFile` using
# :class:`~geoutils.Raster.to_rio_dataset`

rast.to_rio_dataset()
