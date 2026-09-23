"""
Python arithmetic
=================

This example demonstrates arithmetic operations using raster arithmetic on :class:`xarray.DataArray` objects with the :class:`rst <geoutils.RasterAccessor>` accessor. See :ref:`core-py-ops` for more details.
"""

# %%
# We open a raster

# sphinx_gallery_thumbnail_number = 2
import geoutils as gu

filename_rast = gu.examples.get_path("everest_landsat_b4")
rast = gu.open_raster(filename_rast)
rast

# %% We plot the original raster.
rast.rst.plot(cmap="Greys_r")

# %%
# Performing arithmetic operations implicitly loads the data.
rast = (rast + 1.0) ** 0.5 / 5
rast.rst.plot(cmap="Greys_r")

# %%
#
# .. important::
#        Arithmetic operations cast to new :class:`dtypes<numpy.dtype>` automatically following Xarray and NumPy coercion rules.
#
# Logical comparison operations will naturally return a boolean :class:`xarray.DataArray`.

mask = rast == 200
mask

# %%
# Boolean :class:`xarray.DataArray` objects support Python logical operators to be combined together.

mask = (rast >= 3) | (rast % 2 == 0) & (rast != 80)
mask.rst.plot()

# %%
# Finally, boolean rasters can be used for selecting values from a raster.

values = rast.where(mask)
