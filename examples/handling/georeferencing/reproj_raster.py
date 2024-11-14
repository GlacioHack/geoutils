"""
Reproject a raster
==================

This example demonstrates the reprojection of a raster using :func:`geoutils.Raster.reproject`.
"""

# %%
# We open two example rasters.

import geoutils as gu

filename_rast1 = gu.examples.get_path("everest_landsat_b4")
filename_rast2 = gu.examples.get_path("everest_landsat_b4_cropped")
rast1 = gu.Raster(filename_rast1)
rast2 = gu.Raster(filename_rast2)

# %%
# The first raster has larger extent and higher resolution than the second one.
rast1.info()
rast2.info()

# %%
# Let's plot the first raster, with the warped extent of the second one.

rast1.plot(cmap="Blues")
vect_bounds_rast2 = gu.Vector.from_bounds_projected(rast2)
vect_bounds_rast2.plot(fc="none", ec="r", lw=2)

# %%
# **First option:** using the second raster as a reference to match, we reproject the first one. We simply have to pass the second :class:`~geoutils.Raster`
# as single argument to :func:`~geoutils.Raster.reproject`. See :ref:`core-match-ref` for more details.
#
# By default, a "bilinear" resampling algorithm is used. Any string or :class:`~rasterio.enums.Resampling` can be passed.

rast1_warped = rast1.reproject(rast2)
rast1_warped

# %%
# .. note::
#   Because no :attr:`geoutils.Raster.nodata` value is defined in the original image, the default value ``255`` for :class:`numpy.uint8` is used. This
#   value is detected as already existing in the original raster, however, which raises a ``UserWarning``. If your :attr:`geoutils.Raster.nodata` is not defined,
#   use :func:`geoutils.Raster.set_nodata`.
#
# Now the shape and georeferencing should be the same as that of the second raster, shown above.

rast1_warped.info()

# %%
# We can plot the two rasters next to one another

rast1_warped.plot(ax="new", cmap="Reds")
rast2.plot(ax="new", cmap="Blues")

# %%
# **Second option:** we can pass any georeferencing argument to :func:`~geoutils.Raster.reproject`, such as ``dst_size`` and ``dst_crs``, and will only
# deduce other parameters from the raster from which it is called (for ``dst_crs``, an EPSG code can be passed directly as :class:`int`).

# Ensure the right nodata value is set
rast2.set_nodata(0)
# Pass the desired georeferencing parameters
rast2_warped = rast2.reproject(grid_size=(100, 100), crs=32645)
rast2_warped.info()
