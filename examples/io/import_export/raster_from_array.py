"""
Creating a raster from array
============================

This example demonstrates the creation of a raster through :func:`~geoutils.Raster.from_array`.
"""

import numpy as np
import pyproj
import rasterio as rio

# %%
# We create a data array as :class:`~numpy.ndarray`, a transform as :class:`affine.Affine` and a coordinate reference system (CRS) as :class:`pyproj.CRS`.
import geoutils as gu

# A random 3 x 3 masked array
rng = np.random.default_rng(42)
arr = rng.normal(size=(5, 5))
# Introduce a NaN value
arr[2, 2] = np.nan
# A transform with 3 x 3 pixels in a [0-1, 0-1] bound square
transform = rio.transform.from_bounds(0, 0, 1, 1, 3, 3)
# A CRS, here geographic (latitude/longitude)
crs = pyproj.CRS.from_epsg(4326)

# Create a raster
rast = gu.Raster.from_array(data=arr, transform=transform, crs=crs, nodata=255)
rast

# %%
# We can print info on the raster.
rast.info()

# %%
# The array has been automatically cast into a :class:`~numpy.ma.MaskedArray`, to respect :class:`~geoutils.Raster.nodata` values.
rast.data

# %%
# We could also have created directly from a :class:`~numpy.ma.MaskedArray`.

# A random mask, that will mask one out of two values on average
mask = rng.integers(0, 2, size=(5, 5), dtype="bool")
ma = np.ma.masked_array(data=arr, mask=mask)

# This time, we pass directly the masked array
rast = gu.Raster.from_array(data=ma, transform=transform, crs=crs, nodata=255)
rast

# %%
# The different functionalities of GeoUtils will respect :class:`~geoutils.Raster.nodata` values, starting with :func:`~geoutils.Raster.plot`,
# which will ignore them during plotting (transparent).
rast.plot(cmap="copper")
