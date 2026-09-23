"""
Gridding points to raster
=========================

This example demonstrates the gridding of a point cloud into a raster using :meth:`~geoutils.pointcloud.base.PointCloudBase.grid`.
"""

# %%
# We open an example point cloud, an elevation dataset in New Zealand.

# sphinx_gallery_thumbnail_number = 2
import geoutils as gu

filename_pc = gu.examples.get_path("coromandel_lidar")
pc = gu.open_pointcloud(filename_pc, data_column="Z")

# Plot the point cloud
pc.pc.plot(cmap="terrain", cbar_title="Elevation (m)")

# %%
# We generate grid coordinates to interpolate to, alternatively we could pass a raster to use as reference.

import numpy as np

grid_coords = (
    np.linspace(pc.pc.bbox.left, pc.pc.bbox.right, 100),
    np.linspace(pc.pc.bbox.bottom, pc.pc.bbox.top, 100),
)

# %%
# We then perform the interpolation
rast = pc.pc.grid(grid_coords=grid_coords)

# %%
# Finally, we plot the resulting raster

rast.rst.plot(ax="new", cmap="terrain", cbar_title="Elevation (m)")
