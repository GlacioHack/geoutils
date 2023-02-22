"""
Raster to points
================

This example demonstrates the conversion of a raster to point vector using :func:`geoutils.Raster.to_points`.
"""
# sphinx_gallery_thumbnail_number = 2
# %%
# We open a raster.
import geoutils as gu
rast = gu.Raster(gu.examples.get_path("exploradores_aster_dem"))
rast.crop([rast.bounds.left, rast.bounds.bottom, rast.bounds.left+500, rast.bounds.bottom+500])
# %%
# Let's plot the raster.
rast.show(cmap="terrain")

# %%
# We convert the raster to points.

pts_rast = rast.to_points(as_frame=True)
pts_rast.plot(column="b1", cmap="terrain", legend=True)
