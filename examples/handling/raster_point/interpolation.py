"""
Interpolate raster at points
============================

This example demonstrates the 2D interpolation of raster values to points using :meth:`~geoutils.raster.base.RasterBase.interp_at_points`.
"""

# %%
# We open an example raster, a digital elevation model in South America.

# sphinx_gallery_thumbnail_number = 2
import geoutils as gu

filename_rast = gu.examples.get_path("exploradores_aster_dem")
rast = gu.open_raster(filename_rast)
rast = rast.rst.crop([rast.rst.bbox.left, rast.rst.bbox.bottom, rast.rst.bbox.left + 2000, rast.rst.bbox.bottom + 2000])

# Plot the raster
rast.rst.plot(cmap="terrain")

# %%
# We generate a random subsample of 100 coordinates to interpolate.

import numpy as np

rng = np.random.default_rng(42)
x_coords = rng.uniform(rast.rst.bbox.left + 50, rast.rst.bbox.right - 50, 50)
y_coords = rng.uniform(rast.rst.bbox.bottom + 50, rast.rst.bbox.top - 50, 50)

pc = rast.rst.interp_at_points(points=(x_coords, y_coords))

# %%
# We plot the resulting point cloud
pc.pc.plot(ax="new", cmap="terrain", marker="x", cbar_title="Elevation (m)")

# %%
# .. important::
#       Interpolation uses pixel centers independently of the raster's "Area" or "Point" interpretation.

# %%
# We can choose a different interpolation method. The default is "linear".

pc_quintic = rast.rst.interp_at_points(points=(x_coords, y_coords), method="quintic")
np.nanmean(pc.pc.data - pc_quintic.pc.data)
