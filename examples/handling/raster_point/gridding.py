"""
Gridding points to raster
=========================

This example demonstrates the gridding of a point cloud into a raster using :func:`~geoutils.PointCloud.gridding`.
"""

# %%
# We open an example point cloud, an elevation dataset in New Zealand.

# sphinx_gallery_thumbnail_number = 2
import geoutils as gu

filename_pc = gu.examples.get_path("coromandel_lidar")
pc = gu.PointCloud(filename_pc, data_column="Z")

# Plot the point cloud
pc.plot(cmap="terrain", cbar_title="Elevation (m)")

# %%
# We generate grid coordinates to interpolate to, alternatively we could pass a raster to use as reference.

import numpy as np

grid_coords = (np.linspace(pc.bounds.left, pc.bounds.right, 100), np.linspace(pc.bounds.bottom, pc.bounds.top, 100))

# %%
# We then perform the interpolation
rast = pc.grid(grid_coords=grid_coords)

# %%
# Finally, we plot the resulting raster

rast.plot(ax="new", cmap="terrain", cbar_title="Elevation (m)")
