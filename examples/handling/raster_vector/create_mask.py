"""
Mask from a vector
==================

This example demonstrates the creation of a mask from a vector using :meth:`~geoutils.vector.base.VectorBase.create_mask`.
"""

# %%
# We open a raster and vector.

# sphinx_gallery_thumbnail_number = 2
import geoutils as gu

filename_rast = gu.examples.get_path("everest_landsat_b4")
filename_vect = gu.examples.get_path("everest_rgi_outlines")
rast = gu.open_raster(filename_rast)
vect = gu.open_vector(filename_vect)

# %%
# Let's plot the raster and vector.
rast.rst.plot(cmap="Purples")
vect.vct.plot(ref=rast, fc="none", ec="k", lw=2)

# %%
# **First option:** using the raster as a reference to match, we create a mask for the vector in any projection and georeferenced grid. We simply have to pass
# the raster as single argument to :meth:`~geoutils.vector.base.VectorBase.rasterize`. See :ref:`core-match-ref` for more details.

vect_rasterized = vect.vct.create_mask(rast)
vect_rasterized.rst.plot(ax="new")

# %%
# .. note::
#         This is equivalent to using :meth:`~geoutils.vector.base.VectorBase.rasterize` with ``in_value=1`` and ``out_value=0`` and
#         will return a boolean :class:`xarray.DataArray`.

vect_rasterized

# %%
# **Second option:** we can pass any georeferencing parameter to :meth:`~geoutils.vector.base.VectorBase.create_mask`. Any unpassed attribute will be deduced from the
# vector itself, except from the raster shape that will default to 1000 x 1000.


# vect_rasterized = vect.vct.create_mask(res=500)
# vect_rasterized.rst.plot()

# %%
# .. important::
#      The raster shape or resolution are the only unknown arguments to rasterize a vector,
#      one or the other can be passed.
#
