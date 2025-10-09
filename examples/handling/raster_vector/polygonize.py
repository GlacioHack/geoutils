"""
Polygonize a raster
===================

This example demonstrates the polygonizing of a raster using :func:`geoutils.Raster.polygonize`.
"""

# %%
# We open a raster.

# sphinx_gallery_thumbnail_number = 3
import geoutils as gu

filename_rast = gu.examples.get_path("exploradores_aster_dem")
rast = gu.Raster(filename_rast)
rast = rast.crop([rast.bounds.left, rast.bounds.bottom, rast.bounds.left + 5000, rast.bounds.bottom + 5000])
# %%
# Let's plot the raster.
rast.plot(cmap="terrain")

# %%
# We polygonize the raster.

rast_polygonized = rast.polygonize()
rast_polygonized.plot(ax="new")

# %%
# By default, :func:`~geoutils.Raster.polygonize` will try to polygonize target all valid values. Instead, one can specify discrete values to target by
# passing a number or :class:`list`, or a range of values by passing a :class:`tuple`.

# A range of values to polygonize
rast_polygonized = rast.polygonize((2500, 3000))
rast_polygonized.plot(ax="new")

# %%
# An even simpler way to do this is to compute a boolean :func:`~geoutils.Raster` to polygonize using logical
# comparisons on the :func:`~geoutils.Raster`.

rast_polygonized = ((2500 < rast) & (rast < 3000)).polygonize()
rast_polygonized.plot(ax="new")

# %%
# .. note::
#           See :ref:`core-py-ops` for more details on casting to boolean :func:`~geoutils.Raster`.
