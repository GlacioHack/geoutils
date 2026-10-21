"""
Crop a vector
=============

This example demonstrates the cropping of a vector using :meth:`~geoutils.vector.base.VectorBase.crop`.
"""

# %%
# We open a raster and vector.

# sphinx_gallery_thumbnail_number = 3
import geoutils as gu

filename_rast = gu.examples.get_path("everest_landsat_b4_cropped")
filename_vect = gu.examples.get_path("everest_rgi_outlines")
rast = gu.open_raster(filename_rast)
vect = gu.open_vector(filename_vect)

# %%
# Let's plot the raster and vector. The raster has smaller extent than the vector.
rast.rst.plot(cmap="Greys_r", alpha=0.7)
vect.vct.plot(ref=rast, fc="none", ec="tab:purple", lw=3)

# %%
# **First option:** using the raster as a reference to match, we crop the vector. We simply have to pass the raster as single argument to
# :meth:`~geoutils.vector.base.VectorBase.crop`. See :ref:`core-match-ref` for more details.

vect = vect.vct.crop(rast)

# %%
# .. note::
#      :meth:`~geoutils.vector.base.VectorBase.crop` returns a new vector and leaves the source unchanged.
#

rast.rst.plot(ax="new", cmap="Greys_r", alpha=0.7)
vect.vct.plot(ref=rast, fc="none", ec="tab:purple", lw=3)

# %%
# The :meth:`~geoutils.vector.base.VectorBase.crop` keeps all features with geometries intersecting the extent without changing them.
# Use :meth:`~geoutils.vector.base.VectorBase.clip` to cut their geometries exactly at the raster footprint.

vect = vect.vct.clip(rast)
rast.rst.plot(ax="new", cmap="Greys_r", alpha=0.7)
vect.vct.plot(ref=rast, fc="none", ec="tab:purple", lw=3)

# %%
# **Second option:** we can pass other arguments to :meth:`~geoutils.vector.base.VectorBase.crop`, including another vector or a
# simple :class:`tuple` of bounds.

bbox = rast.rst.get_bbox_projected(out_crs=vect.vct.crs)
vect = vect.vct.crop((bbox.left + 0.5 * (bbox.right - bbox.left), bbox.bottom, bbox.right, bbox.top))

rast.rst.plot(ax="new", cmap="Greys_r", alpha=0.7)
vect.vct.plot(ref=rast, fc="none", ec="tab:purple", lw=3)
