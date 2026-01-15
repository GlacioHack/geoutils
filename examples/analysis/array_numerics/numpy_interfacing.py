"""
NumPy interfacing
=================

This example demonstrates NumPy interfacing with rasters on :class:`Rasters<geoutils.Raster>`. See :ref:`core-array-funcs` for more details.
"""

# %%
# We open a raster.

# sphinx_gallery_thumbnail_number = 2
import geoutils as gu

filename_rast = gu.examples.get_path("exploradores_aster_dem")
rast = gu.Raster(filename_rast)

# %% We plot it.
rast.plot(cmap="terrain")

# %%
#
# The NumPy interface allows to use almost any NumPy function directly on the raster.

import numpy as np

# Get the x and y gradient as 1D arrays
gradient_y, gradient_x = np.gradient(rast)
# Estimate the orientation in degrees casting to 2D
aspect = np.arctan2(-gradient_x, gradient_y)
aspect = (aspect * 180 / np.pi) + np.pi

aspect.plot(cmap="twilight", cbar_title="Aspect (degrees)")

# %%
#
# .. important::
#        For rigorous slope and aspect calculation (matching that of GDAL), **check-out our sister package** `xDEM <https://xdem.readthedocs.io/en/latest/index.html>`_.
#
# We use NumPy logical operations to isolate the terrain oriented South and above three thousand meters. The rasters will be logically cast to a
# boolean :class:`Raster<geoutils.Raster>`.

mask = np.logical_and.reduce((aspect > -45, aspect < 45, rast > 3000))
mask

# %%
# We plot the mask.

mask.plot()
