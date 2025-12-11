"""
Open/save a point cloud
=======================

This example demonstrates the instantiation of a point cloud through :class:`geoutils.PointCloud` and saving with :func:`~geoutils.Vector.save`.
"""

import geoutils as gu

# %%
# We open an example vector.
filename_pc = gu.examples.get_path("coromandel_lidar")
pc = gu.PointCloud(filename_pc, data_column="Z")
pc

# %%
# A point cloud is a subclass of :class:`~geoutils.Vector`, with a main attribute :attr:`~geoutils.PointCloud.data_column` pointing to the main data column
# of the point cloud.
# All other attributes are :ref:`inherited from Shapely and GeoPandas<vector-from-geopandas>`. See also :ref:`vector-class`.

# %%
#
# .. note::
#        A point cloud can also be instantiated with a :class:`geopandas.GeoDataFrame`, see :ref:`sphx_glr_io_examples_import_export_import_vector.py`.
#
# We can print more info on the point cloud.
pc.info()

# %%
# Let's plot the point cloud main column
pc.plot(cbar_title="Elevation (m)")

# %%
# Finally, a point cloud is saved using :func:`~geoutils.Vector.save`.

pc.to_file("mypc.gpkg")
