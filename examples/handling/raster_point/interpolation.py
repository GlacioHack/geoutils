"""
Interpolate raster at points
============================

This example demonstrates the 2D interpolation of raster values to points using :func:`~geoutils.Raster.interp_points`.
"""

# %%
# We open an example raster, a digital elevation model in South America.

# sphinx_gallery_thumbnail_number = 2
import geoutils as gu

filename_rast = gu.examples.get_path("exploradores_aster_dem")
rast = gu.Raster(filename_rast)
rast = rast.crop([rast.bounds.left, rast.bounds.bottom, rast.bounds.left + 2000, rast.bounds.bottom + 2000])

# Plot the raster
rast.plot(cmap="terrain")

# %%
# We generate a random subsample of 100 coordinates to interpolate.

import numpy as np

rng = np.random.default_rng(42)
x_coords = rng.uniform(rast.bounds.left + 50, rast.bounds.right - 50, 50)
y_coords = rng.uniform(rast.bounds.bottom + 50, rast.bounds.top - 50, 50)

pc = rast.interp_points(points=(x_coords, y_coords))

# %%
# We plot the resulting point cloud
pc.plot(ax="new", cmap="terrain", marker="x", cbar_title="Elevation (m)")

# %%
# .. important::
#       The interpretation of where raster values are located can differ. The parameter ``shift_area_or_point`` (off by default) can be turned on to ensure
#       that the pixel interpretation of your dataset is correct.

# %%
# Let's look and redefine our pixel interpretation into ``"Point"``. This will shift interpolation by half a pixel.

rast.area_or_point
rast.area_or_point = "Point"

# %%
# We can interpolate again by shifting according to our interpretation, and changing the resampling algorithm (default to "linear").

pc_shifted = rast.interp_points(points=(x_coords, y_coords), shift_area_or_point=True, method="quintic")
np.nanmean(pc - pc_shifted)

# %%
# The mean difference in interpolated values is quite significant, with a 2-meter bias!
