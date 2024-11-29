"""
Raster to points
================

This example demonstrates the conversion of a raster to point vector using :func:`geoutils.Raster.to_points`.
"""

# %%
# We open a raster.

# sphinx_gallery_thumbnail_number = 2
import geoutils as gu

filename_rast = gu.examples.get_path("exploradores_aster_dem")
rast = gu.Raster(filename_rast)
rast = rast.crop([rast.bounds.left, rast.bounds.bottom, rast.bounds.left + 500, rast.bounds.bottom + 500])

# %%
# Let's plot the raster.
rast.plot(cmap="terrain")

# %%
# We convert the raster to points. By default, this returns a vector with column geometry burned.

pts_rast = rast.to_pointcloud()
pts_rast

# %%
# We plot the point vector.

pts_rast.plot(ax="new", column="b1", cmap="terrain", legend=True)
