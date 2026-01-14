"""
Open/save a vector
==================

This example demonstrates the instantiation of a vector through :class:`geoutils.Vector` and saving with :func:`~geoutils.Vector.save`.
"""

import geoutils as gu

# %%
# We open an example vector.
filename_vect = gu.examples.get_path("everest_rgi_outlines")
vect = gu.Vector(filename_vect)
vect

# %%
# A vector is composed of a single main attribute: a :class:`~geoutils.Vector.ds` geodataframe.
# All other attributes are :ref:`inherited from Shapely and GeoPandas<vector-from-geopandas>`. See also :ref:`vector-class`.

# %%
#
# .. note::
#        A vector can also be instantiated with a :class:`geopandas.GeoDataFrame`, see :ref:`sphx_glr_io_examples_import_export_import_vector.py`.
#
# We can print more info on the vector.
vect.info()

# %%
# Let's plot by vector area
vect.plot(column="Area", cbar_title="Area (km²)")

# %%
# Finally, a vector is saved using :func:`~geoutils.Vector.save`.

vect.to_file("myvector.gpkg")
