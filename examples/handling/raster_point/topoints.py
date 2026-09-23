"""
Raster to regular points
========================

This example demonstrates the conversion of raster regular-grid values to a point cloud using :meth:`~geoutils.raster.base.RasterBase.to_pointcloud`.
"""

# %%
# We open a raster.

# sphinx_gallery_thumbnail_number = 2
import geoutils as gu

filename_rast = gu.examples.get_path("exploradores_aster_dem")
rast = gu.open_raster(filename_rast)
rast = rast.rst.crop([rast.rst.bbox.left, rast.rst.bbox.bottom, rast.rst.bbox.left + 500, rast.rst.bbox.bottom + 500])

# %%
# Let's plot the raster.
rast.rst.plot(cmap="terrain")

# %%
# We convert the raster to points. By default, this returns a vector with column geometry burned.

pc = rast.rst.to_pointcloud()
pc

# %%
# We plot the point vector.

pc.pc.plot(ax="new", cmap="terrain", legend=True)
