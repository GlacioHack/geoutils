(api-pc-dataset)=
# Dataset.pc accessor

The {class}`Dataset.pc <geoutils.DatasetPointCloudAccessor>` accessor exposes GeoUtils point cloud methods on {class}`xarray.Dataset` objects.

It applies operations through {class}`DataArray.pc <geoutils.DataArrayPointCloudAccessor>` to all compatible point variables
by default, or to those selected with `variables`.

**For the main point cloud API reference page, see {ref}`pointcloud-api`.**
Below is the full class description specific to {class}`Dataset.pc <geoutils.DatasetPointCloudAccessor>`:

```{eval-rst}
.. currentmodule:: geoutils

.. autoclass:: DatasetPointCloudAccessor
    :members:
    :undoc-members: false
    :show-inheritance:
```
