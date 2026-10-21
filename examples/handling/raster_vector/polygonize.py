"""
Polygonize a raster
===================

This example demonstrates the polygonizing of a raster using :meth:`~geoutils.raster.base.RasterBase.polygonize`.
"""

# %%
# We open a raster.

# sphinx_gallery_thumbnail_number = 3
import geoutils as gu

filename_rast = gu.examples.get_path("exploradores_aster_dem")
rast = gu.open_raster(filename_rast)
rast = rast.rst.crop([rast.rst.bbox.left, rast.rst.bbox.bottom, rast.rst.bbox.left + 5000, rast.rst.bbox.bottom + 5000])
# %%
# Let's plot the raster.
rast.rst.plot(cmap="terrain")

# %%
# We polygonize the raster.

rast_polygonized = rast.rst.polygonize()
rast_polygonized.vct.plot(ax="new")

# %%
# By default, :meth:`~geoutils.raster.base.RasterBase.polygonize` will try to polygonize target all valid values. Instead, one can specify discrete values to target by
# passing a number or :class:`list`, or a range of values by passing a :class:`tuple`.

# A range of values to polygonize
rast_polygonized = rast.rst.polygonize((2500, 3000))
rast_polygonized.vct.plot(ax="new")

# %%
# An even simpler way to do this is to compute a boolean :class:`xarray.DataArray` to polygonize using logical
# comparisons on the raster.

rast_polygonized = ((2500 < rast) & (rast < 3000)).rst.polygonize()
rast_polygonized.vct.plot(ax="new")

# %%
# .. note::
#           See :ref:`core-py-ops` for more details on casting to boolean.
