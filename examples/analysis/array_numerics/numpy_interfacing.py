"""
NumPy interfacing
=================

This example demonstrates NumPy interfacing with rasters on :class:`Rasters<geoutils.Raster>`. See :ref:`core-array-funcs` for more details.
"""
# sphinx_gallery_thumbnail_number = 2
# %%
# We open a raster
import geoutils as gu
rast = gu.Raster(gu.examples.get_path("exploradores_aster_dem"))

# %% We plot the original raster.
rast.show(cmap='terrain')

# %%
# NumPy interfacing allows to use any NumPy function directly on the raster

import numpy as np

# Get the x and y gradient as 1D arrays
gradient_y, gradient_x = np.gradient(rast)
# Estimate the orientation in degrees casting to 2D
aspect = np.arctan2(-gradient_x, gradient_y)
aspect = (aspect * 180 / np.pi) + np.pi
# We copy the new array into a raster
asp = rast.copy(new_array=aspect)

asp.show(cmap='twilight', cbar_title="Aspect (degrees)")

# %%
#
# .. important::
#        For rigorous slope and aspect calculation (matching that of GDAL), **check-out our sister package** `xDEM <https://xdem.readthedocs.io/en/latest/index.html>`_.
#
# We can make numpy logical operations to isolate the terrain oriented South and above three thousand meters. The rasters will be cast to a :class:`Mask<geoutils.Mask>`.

# Not supported yet, fix first
# mask = np.logical_and.reduce((asp > -45, asp < 45, rast > 3000))
# mask

# %%
# We plot the mask.

# mask.show()


