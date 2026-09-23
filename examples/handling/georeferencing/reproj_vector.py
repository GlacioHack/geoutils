"""
Reproject a vector
==================

This example demonstrates the reprojection of a vector using :meth:`~geoutils.vector.base.VectorBase.reproject`.
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
# The two objects are in different projections.
rast.rst.info()
vect.vct.info()

# %%
# Let's plot the two in their original projection.
rast.rst.plot(cmap="Greys_r")
vect.vct.plot(ax="new", fc="none", ec="tab:purple", lw=3)

# %%
# **First option:** using the raster as a reference to match, we reproject the vector. We simply have to pass the raster as an argument
# to :meth:`~geoutils.vector.base.VectorBase.reproject`. See :ref:`core-match-ref` for more details.

vect_reproj = vect.vct.reproject(rast)

# %%
# We can plot the vector in its new projection.

vect_reproj.vct.plot(ax="new", fc="none", ec="tab:purple", lw=3)

# %%
# **Second option:** we can pass the georeferencing argument ``dst_crs`` to :meth:`~geoutils.vector.base.VectorBase.reproject` (an EPSG code can be passed directly as
# :class:`int`).

# Reproject in UTM zone 45N.
vect_reproj = vect.vct.reproject(crs=32645)
vect_reproj
