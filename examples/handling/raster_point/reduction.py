"""
Reduce raster around points
===========================

This example demonstrates the reduction of windowed raster values around a point using :meth:`~geoutils.raster.base.RasterBase.reduce_points`.
"""

# %%
# We open an example raster, a digital elevation model in South America.

# sphinx_gallery_thumbnail_number = 3
import geoutils as gu

filename_rast = gu.examples.get_path("exploradores_aster_dem")
rast = gu.open_raster(filename_rast)
rast = rast.rst.crop([rast.rst.bbox.left, rast.rst.bbox.bottom, rast.rst.bbox.left + 2000, rast.rst.bbox.bottom + 2000])

# Plot the raster
rast.rst.plot(cmap="terrain")

# %%
# We generate a random subsample of 100 coordinates to extract.

import geopandas as gpd
import numpy as np

# Replace by Raster function once done
rng = np.random.default_rng(42)
x_coords = rng.uniform(rast.rst.bbox.left + 50, rast.rst.bbox.right - 50, 50)
y_coords = rng.uniform(rast.rst.bbox.bottom + 50, rast.rst.bbox.top - 50, 50)

pc = rast.rst.reduce_points((x_coords, y_coords))

# %%
# We plot the resulting point cloud
pc.pc.plot(ax="new", cmap="terrain", cbar_title="Elevation (m)")

# %%
# By default, :meth:`~geoutils.raster.base.RasterBase.reduce_points` extracts the closest pixel value. But it can also be passed a window size and reductor function to
# extract an average value or other statistic based on neighbouring pixels.

pc_reduced = rast.rst.reduce_points((x_coords, y_coords), window=5, reducer_function=np.nanmedian)

np.nanmean(pc.pc.data - pc_reduced.pc.data)

# %%
# The mean difference in extracted values is quite significant at 0.3 meters!
# We can visualize how the sampling took place in the windows.

# Replace by Vector function once done
coords = rast.rst.coords(grid=True)
x_closest = rast.rst.copy(new_array=coords[0]).rst.reduce_points((x_coords, y_coords), as_array=True).squeeze()
y_closest = rast.rst.copy(new_array=coords[1]).rst.reduce_points((x_coords, y_coords), as_array=True).squeeze()
from shapely.geometry import box

geometry = [
    box(x - 2 * rast.rst.res[0], y - 2 * rast.rst.res[1], x + 2 * rast.rst.res[0], y + 2 * rast.rst.res[1])
    for x, y in zip(x_closest, y_closest)
]
ds = gpd.GeoDataFrame(geometry=geometry, crs=rast.rst.crs)
ds["vals"] = pc_reduced.pc.data
ds.plot(column="vals", cmap="terrain", legend=True, vmin=np.nanmin(rast), vmax=np.nanmax(rast))
