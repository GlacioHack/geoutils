"""
Open/save a point cloud
=======================

This example demonstrates opening a point cloud with :func:`geoutils.open_pointcloud` and saving it with the :class:`pc <geoutils.PointCloudAccessor>` accessor.
"""

import geoutils as gu

# %%
# We open an example vector.
filename_pc = gu.examples.get_path("coromandel_lidar")
pc = gu.open_pointcloud(filename_pc, data_column="Z")
pc

# %%
# A point cloud is a :class:`geopandas.GeoDataFrame`, with a main attribute :attr:`~geoutils.pointcloud.base.PointCloudBase.data_column` pointing to the main data column
# through its ``pc`` accessor.
# All other attributes are :ref:`inherited from Shapely and GeoPandas<vector-from-geopandas>`. See also :ref:`vector-class`.

# %%
#
# .. note::
#        GeoUtils point cloud methods can also be used on an existing :class:`geopandas.GeoDataFrame`.
#
# We can print more info on the point cloud.
pc.pc.info()

# %%
# Let's plot the point cloud main column
pc.pc.plot(cbar_title="Elevation (m)")

# %%
# Finally, a point cloud is saved using :meth:`~geoutils.VectorAccessor.to_file`.

pc.pc.to_file("mypc.gpkg")
