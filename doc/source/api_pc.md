(api-pc)=
# GeoDataFrame.pc accessor

The {class}`pc <geoutils.GeoDataFramePointCloudAccessor>` accessor exposes GeoUtils point cloud methods on {class}`geopandas.GeoDataFrame` objects.

It mirrors the API of {class}`~geoutils.PointCloud`; native GeoPandas methods remain available directly on the dataframe.

**For the main point cloud API reference page, see {ref}`pointcloud-api`.**
Below is the full class description specific to {class}`pc <geoutils.GeoDataFramePointCloudAccessor>`:

```{eval-rst}
.. currentmodule:: geoutils

.. autoclass:: GeoDataFramePointCloudAccessor
    :members:
    :exclude-members: to_file
    :inherited-members:
    :undoc-members: false
    :show-inheritance:
```
