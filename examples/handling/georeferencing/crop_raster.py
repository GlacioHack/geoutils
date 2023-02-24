"""
Crop a raster
=============

This example demonstrates the cropping of a raster using :func:`geoutils.Raster.crop`.
"""
# sphinx_gallery_thumbnail_number = 2
# %%
# We open a raster and vector, and subset the latter.
import geoutils as gu
rast = gu.Raster(gu.examples.get_path("everest_landsat_b4"))
vect = gu.Vector(gu.examples.get_path("everest_rgi_outlines"))
vect = gu.Vector(vect.ds[vect.ds["RGIId"] == "RGI60-15.10055"])

# %%
# The first raster has larger extent and higher resolution than the second one.
print(rast.info())
print(vect.bounds)

# %%
# Let's plot the raster and vector.
import matplotlib.pyplot as plt
rast.show(cmap="Purples")
vect.show(ref_crs=rast, fc='none', ec='k', lw=2)

# %%
# **First option:** using the second raster as a reference to match, we reproject the first one. We simply have to pass the second :class:`~geoutils.Raster`
# as single argument to :func:`~geoutils.Raster.crop`. See :ref:`core-match-ref` for more details.

rast.crop(vect)

# %%
# Now the bounds should be the same as that of the vector (within the size of a pixel as the grid was not warped).
#
# .. note::
#      By default, :func:`~geoutils.Raster.crop` is done in-place, replacing ``rast``. This behaviour can be modified by passing ``inplace=False``.
#

rast.show(ax="new", cmap="Purples")
vect.show(ref_crs=rast, fc='none', ec='k', lw=2)

# %%
# **Second option:** we can pass other ``crop_geom`` argument to :func:`~geoutils.Raster.crop`, including another :class:`~geoutils.Raster` or a
# simple :class:`tuple` of bounds.

rast.crop((rast.bounds.left + 1000, rast.bounds.bottom, rast.bounds.right, rast.bounds.top - 500))

rast.show(ax="new", cmap="Purples")
vect.show(ref_crs=rast, fc='none', ec='k', lw=2)
