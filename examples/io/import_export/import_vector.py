"""
From/To GeoPandas
=================

This example demonstrates importing or exporting a :class:`geopandas.GeoDataFrame` from and to a :class:`~geoutils.Vector`.
"""

# %%
# A vector can be imported from a :class:`geopandas.GeoDataFrame` simply by instantiating :class:`~geoutils.Vector`.

import geoutils as gu
import geopandas as gpd

ds = gpd.read_file(gu.examples.get_path("everest_rgi_outlines"))
vect = gu.Vector(ds)
vect

# %%
# To export, the :class:`geopandas.GeoDataFrame` is always stored as an attribute as :class:`~geoutils.Vector` is composed from it. See :ref:`core-composition`.

vect.ds
