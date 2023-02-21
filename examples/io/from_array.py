"""
Creating a raster from array
============================

This example demonstrates the creation of a raster through :func:`~geoutils.Raster.from_array`.
"""

# %%
# We create a data array as :class:`~numpy.ndarray`, a transform as :class:`affine.Affine` and a coordinate reference system (CRS) as :class:`pyproj.CRS`.
import geoutils as gu
import rasterio as rio
import pyproj
import numpy as np

# A random 3 x 3 masked array
np.random.seed(42)
arr = np.random.randint(0, 255, size=(3, 3), dtype="float32")
# A transform with 3 x 3 pixels in a [0-1, 0-1] bound square
transform = rio.transform.from_bounds(0, 0, 1, 1, 3, 3)
# A CRS, here geographic (latitude/longitude)
crs = pyproj.CRS.from_epsg(4326)

# Create a raster
rast = gu.Raster.from_array(
        data = arr,
        transform = transform,
        crs = crs,
        nodata = 255
    )
rast

# %%
# We can print info on the raster.
print(rast.info())

# %%
# The array has been automatically cast into a :class:`~numpy.ma.MaskedArray`, to respect :class:`~geoutils.Raster.nodata` values.

# %%
# We could also have created directly from a :class:`~numpy.ma.MaskedArray`.

# A random mask, that will mask one out of two values on average
mask = np.random.randint(0, 2, size=(3, 3), dtype="bool")
ma = np.ma.masked_array(data=arr, mask=mask)

rast = gu.Raster.from_array(
        data = arr,
        transform = transform,
        crs = crs,
        nodata = 255
    )
rast

# %%
# The different functionalities of GeoUtils will respect :class:`~geoutils.Raster.nodata` values, starting with :func:`~geoutils.Raster.show`.
rast.show(cmap="Greys_r")
