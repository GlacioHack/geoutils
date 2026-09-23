"""
Open/save a vector
==================

This example demonstrates opening a vector with :func:`geoutils.open_vector` and saving it with the :class:`vct <geoutils.VectorAccessor>` accessor.
"""

import geoutils as gu

# %%
# We open an example vector.
filename_vect = gu.examples.get_path("everest_rgi_outlines")
vect = gu.open_vector(filename_vect)
vect

# %%
# A vector opened through the accessor is a :class:`geopandas.GeoDataFrame`.
# All native attributes are inherited from Shapely and GeoPandas. See also :ref:`vector-class`.

# %%
#
# .. note::
#        GeoUtils methods can also be used on an existing :class:`geopandas.GeoDataFrame`, see :ref:`sphx_glr_io_examples_import_export_import_vector.py`.
#
# We can print more info on the vector.
vect.vct.info()

# %%
# Let's plot by vector area
vect.vct.plot(column="Area", cbar_title="Area (km²)")

# %%
# Finally, a vector is saved using :meth:`~geoutils.VectorAccessor.to_file`.

vect.vct.to_file("myvector.gpkg")
