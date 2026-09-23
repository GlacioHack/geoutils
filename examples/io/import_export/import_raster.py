"""
From/to Rasterio
================

This example demonstrates importing and exporting a :class:`rasterio.io.DatasetReader` or :class:`rasterio.io.MemoryFile` from and to an
:class:`xarray.DataArray` with the :class:`rst <geoutils.RasterAccessor>` accessor.
"""

import rasterio as rio

# %%
# A raster can be imported from a :class:`rasterio.io.DatasetReader` with :func:`~geoutils.open_raster`.
import geoutils as gu

filename_rast = gu.examples.get_path("exploradores_aster_dem")
ds = rio.open(filename_rast)
rast = gu.open_raster(ds)
rast

# %%
# The data is unloaded, as when opened with a filename.
# The data will be loaded explicitly by any function requiring its values, such as plotting.
rast.rst.plot(cmap="terrain")

# %%
# We can also open a :class:`rasterio.io.MemoryFile`.

mem = rio.MemoryFile(open(filename_rast, "rb"))
rast = gu.open_raster(mem)
rast

# %%
# The raster has the same DataArray representation.
#
# Finally, we can export the :class:`xarray.DataArray` to a :class:`rasterio.io.MemoryFile` and open it
# as a :class:`rasterio.io.DatasetReader`.

with rio.MemoryFile() as output_memory:
    rast.rst.to_file(output_memory.name)
    with output_memory.open() as output_dataset:
        print(output_dataset.profile)
