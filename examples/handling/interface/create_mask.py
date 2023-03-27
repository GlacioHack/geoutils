"""
Mask from a vector
==================

This example demonstrates the creation of a mask from a vector using :func:`geoutils.Vector.create_mask`.
"""
# %%
# We open a raster and vector.

# sphinx_gallery_thumbnail_number = 2
import geoutils as gu

rast = gu.Raster(gu.examples.get_path("everest_landsat_b4"))
vect = gu.Vector(gu.examples.get_path("everest_rgi_outlines"))

# %%
# Let's plot the raster and vector.
rast.show(cmap="Purples")
vect.show(ref_crs=rast, fc="none", ec="k", lw=2)

# %%
# **First option:** using the raster as a reference to match, we create a mask for the vector in any projection and georeferenced grid. We simply have to pass
# the :class:`~geoutils.Raster` as single argument to :func:`~geoutils.Vector.rasterize`. See :ref:`core-match-ref` for more details.

vect_rasterized = vect.create_mask(rast)
vect_rasterized.show(ax="new")

# %%
# .. note::
#         This is equivalent to using :func:`~geoutils.Vector.rasterize` with ``in_value=1`` and ``out_value=0`` and will return a :class:`~geoutils.Mask`.

vect_rasterized

# %%
# **Second option:** we can pass any georeferencing parameter to :func:`~geoutils.Raster.create_mask`. Any unpassed attribute will be deduced from the
# :class:`~geoutils.Vector` itself, except from the :attr:`~geoutils.Raster.shape` to rasterize that will default to 1000 x 1000.


# vect_rasterized = vect.create_mask(xres=500)
# vect_rasterized.show()

# %%
# .. important::
#      The :attr:`~geoutils.Raster.shape` or the :attr:`~geoutils.Raster.res` are the only unknown arguments to rasterize a :class:`~geoutils.Vector`,
#      one or the other can be passed.
#
