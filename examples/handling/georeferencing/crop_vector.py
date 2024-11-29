"""
Crop a vector
=============

This example demonstrates the cropping of a vector using :func:`geoutils.Vector.crop`.
"""

# %%
# We open a raster and vector.

# sphinx_gallery_thumbnail_number = 3
import geoutils as gu

filename_rast = gu.examples.get_path("everest_landsat_b4_cropped")
filename_vect = gu.examples.get_path("everest_rgi_outlines")
rast = gu.Raster(filename_rast)
vect = gu.Vector(filename_vect)

# %%
# Let's plot the raster and vector. The raster has smaller extent than the vector.
rast.plot(cmap="Greys_r", alpha=0.7)
vect.plot(ref_crs=rast, fc="none", ec="tab:purple", lw=3)

# %%
# **First option:** using the raster as a reference to match, we crop the vector. We simply have to pass the :class:`~geoutils.Raster` as single argument to
# :func:`~geoutils.Vector.crop`. See :ref:`core-match-ref` for more details.

vect = vect.crop(rast)

# %%
# .. note::
#      By default, :func:`~geoutils.Vector.crop` is done in-place, replacing ``vect``. This behaviour can be modified by passing ``inplace=False``.
#

rast.plot(ax="new", cmap="Greys_r", alpha=0.7)
vect.plot(ref_crs=rast, fc="none", ec="tab:purple", lw=3)

# %%
# The :func:`~geoutils.Vector.crop` keeps all features with geometries intersecting the extent to crop to. We can also force a clipping of the geometries
# within the bounds using ``clip=True``.

vect = vect.crop(rast, clip=True)
rast.plot(ax="new", cmap="Greys_r", alpha=0.7)
vect.plot(ref_crs=rast, fc="none", ec="tab:purple", lw=3)

# %%
# **Second option:** we can pass other ``crop_geom`` argument to :func:`~geoutils.Vector.crop`, including another :class:`~geoutils.Vector` or a
# simple :class:`tuple` of bounds.

bounds = rast.get_bounds_projected(out_crs=vect.crs)
vect = vect.crop(crop_geom=(bounds.left + 0.5 * (bounds.right - bounds.left), bounds.bottom, bounds.right, bounds.top))

rast.plot(ax="new", cmap="Greys_r", alpha=0.7)
vect.plot(ref_crs=rast, fc="none", ec="tab:purple", lw=3)
