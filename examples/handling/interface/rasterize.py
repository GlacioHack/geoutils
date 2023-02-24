"""
Rasterize a vector
==================

This example demonstrates the rasterizing of a vector using :func:`geoutils.Vector.rasterize`.
"""
# sphinx_gallery_thumbnail_number = 2
# %%
# We open a raster and vector.
import geoutils as gu
rast = gu.Raster(gu.examples.get_path("everest_landsat_b4"))
vect = gu.Vector(gu.examples.get_path("everest_rgi_outlines"))

# %%
# Let's plot the raster and vector.
rast.show(cmap="Purples")
vect.show(ref_crs=rast, fc='none', ec='k', lw=2)

# %%
# **First option:** using the raster as a reference to match, we rasterize the vector in any projection and georeferenced grid. We simply have to pass the
# :class:`~geoutils.Raster` as single argument to :func:`~geoutils.Vector.rasterize`. See :ref:`core-match-ref` for more details.

vect_rasterized = vect.rasterize(rast)
vect_rasterized.show(ax="new", cmap="viridis")

# %%
# By default, :func:`~geoutils.Vector.rasterize` will burn the index of the :class:`~geoutils.Vector`'s features in their geometry. We can specify the ``in_value`` to burn a
# single value, or any iterable with the same length as there are features in the :class:`~geoutils.Vector`. An ``out_value`` can be passed to burn
# outside the geometries.
#

vect_rasterized = vect.rasterize(rast, in_value=1)
vect_rasterized.show(ax="new")

# %%
#
# .. note::
#         If the rasterized ``in_value`` is fixed to 1 and ``out_value`` to 0 (default), then :func:`~geoutils.Vector.rasterize` is creating a boolean mask.
#         This is equivalent to using :func:`~geoutils.Vector.create_mask`, and will return a :class:`~geoutils.Mask`.

vect_rasterized

# %%
# **Second option:** we can pass any georeferencing parameter to :func:`~geoutils.Raster.rasterize`. Any unpassed attribute will be deduced from the
# :class:`~geoutils.Vector` itself, except from the :attr:`~geoutils.Raster.shape` to rasterize that will default to 1000 x 1000.


# vect_rasterized = vect.rasterize(xres=500)
# vect_rasterized.show()

# %%
# .. important::
#      The :attr:`~geoutils.Raster.shape` or the :attr:`~geoutils.Raster.res` are the only unknown arguments to rasterize a :class:`~geoutils.Vector`,
#      one or the other can be passed.
#
