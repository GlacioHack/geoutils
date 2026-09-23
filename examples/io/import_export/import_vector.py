"""
From/to GeoPandas
=================

This example demonstrates using the :class:`vct <geoutils.VectorAccessor>` accessor on a :class:`geopandas.GeoDataFrame`.
"""

# %%
# GeoUtils vector methods are available directly on a :class:`geopandas.GeoDataFrame` through its ``vct`` accessor.

import geopandas as gpd

import geoutils as gu

filename_vect = gu.examples.get_path("exploradores_rgi_outlines")
ds = gpd.read_file(filename_vect)
vect = ds
vect

# %%
# We plot the vector.

vect.vct.plot(column="RGIId", add_cbar=False)

# %%
# The vector remains a native :class:`geopandas.GeoDataFrame`, so it can be exported or passed directly to GeoPandas operations.

vect
