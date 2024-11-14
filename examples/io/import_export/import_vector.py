"""
From/to GeoPandas
=================

This example demonstrates importing or exporting a :class:`geopandas.GeoDataFrame` from and to a :class:`~geoutils.Vector`.
"""

# %%
# A vector can be imported from a :class:`geopandas.GeoDataFrame` simply by instantiating :class:`~geoutils.Vector`.

import geopandas as gpd

import geoutils as gu

filename_vect = gu.examples.get_path("exploradores_rgi_outlines")
ds = gpd.read_file(filename_vect)
vect = gu.Vector(ds)
vect

# %%
# We plot the vector.

vect.plot(column="RGIId", add_cbar=False)

# %%
# To export, the :class:`geopandas.GeoDataFrame` is always stored as an attribute as :class:`~geoutils.Vector` is composed from it. See :ref:`core-composition`.

vect.ds
