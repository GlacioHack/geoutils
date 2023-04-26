"""
Interpolation from grid
=======================

This example demonstrates the 2D interpolation of raster values to points using :func:`~geoutils.Raster.interp_points`.
"""
# %%
# We open an example raster, a digital elevation model in South America.

# sphinx_gallery_thumbnail_number = 2
import geoutils as gu

filename_rast = gu.examples.get_path("exploradores_aster_dem")
rast = gu.Raster(filename_rast)
rast.crop([rast.bounds.left, rast.bounds.bottom, rast.bounds.left + 2000, rast.bounds.bottom + 2000])

# Plot the raster
rast.show(cmap="terrain")

# %%
# We generate a random subsample of 100 points to interpolate, and extract the coordinates.

import geopandas as gpd
import numpy as np

# Replace by Raster function once done (valid coords)
np.random.seed(42)
x_coords = np.random.uniform(rast.bounds.left + 50, rast.bounds.right - 50, 50)
y_coords = np.random.uniform(rast.bounds.bottom + 50, rast.bounds.top - 50, 50)

vals = rast.interp_points(pts=list(zip(x_coords, y_coords)))

# %%
# Replace by Vector function once done
ds = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x=x_coords, y=y_coords), crs=rast.crs)
ds["vals"] = vals
ds.plot(column="vals", cmap="terrain", legend=True, vmin=np.nanmin(rast), vmax=np.nanmax(rast), marker="x")

# %%
# .. important::
#       The interpretation of where raster values are located differ, see XXX. The parameter ``shift_area_or_point`` (off by default) can be turned on to ensure
#       that the pixel interpretation of your dataset is correct.

# %%
# Let's look and redefine our pixel interpretation into ``"Point"``. This will shift interpolation by half a pixel.

print(rast.tags["AREA_OR_POINT"])
rast.tags["AREA_OR_POINT"] = "Point"

# %%
# We can interpolate again by shifting according to our interpretation, and changing the resampling algorithm (default to "linear").

vals_shifted = rast.interp_points(pts=list(zip(x_coords, y_coords)), shift_area_or_point=True, mode="quintic")
np.nanmean(vals - vals_shifted)

# %%
# The mean difference in interpolated values is quite significant, with a 2 meter bias!
