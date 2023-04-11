"""
Python arithmetic
=================

This example demonstrates arithmetic operations using raster arithmetic on :class:`Rasters<geoutils.Raster>`. See :ref:`core-py-ops` for more details.
"""
# %%
# We open a raster

# sphinx_gallery_thumbnail_number = 2
import geoutils as gu

filename_rast = gu.examples.get_path("everest_landsat_b4")
rast = gu.Raster(filename_rast)
rast

# %% We plot the original raster.
rast.show(cmap="Greys_r")

# %%
# Performing arithmetic operations implicitly loads the data.
rast = (rast + 1.0) ** 0.5 / 5
rast.show(cmap="Greys_r")

# %%
#
# .. important::
#        Arithmetic operations cast to new :class:`dtypes<numpy.dtype>` automatically following NumPy coercion rules. If we had written ``(rast + 1)``,
#        this calculation would have conserved the original :class:`numpy.uint8` :class:`dtype<numpy.dtype>` of the raster.
#
# Logical comparison operations will naturally cast to a :class:`Mask<geoutils.Mask>`.

mask = rast == 200
mask

# %%
# :class:`Masks<geoutils.Mask>` support python logical operators to be combined together

mask = (rast >= 3) | (rast % 2 == 0) & (rast != 80)
mask.show()

# %%
# Finally, :class:`Masks<geoutils.Mask>` can be used for indexing and assigning to a :class:`Rasters<geoutils.Raster>`

values = rast[mask]
