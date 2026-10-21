(api-vct)=
# GeoDataFrame.vct accessor

The {class}`vct <geoutils.GeoDataFrameVectorAccessor>` accessor exposes GeoUtils vector methods on {class}`geopandas.GeoDataFrame` objects.

It mirrors the GeoUtils-specific API of {class}`~geoutils.Vector`; native GeoPandas methods remain available directly on the dataframe.

**For the main vector API reference page, see {ref}`vector-api`.**
Below is the full class description specific to {class}`vct <geoutils.GeoDataFrameVectorAccessor>`:

```{eval-rst}
.. currentmodule:: geoutils

.. autoclass:: GeoDataFrameVectorAccessor
    :members:
    :exclude-members: to_file
    :inherited-members:
    :undoc-members: false
    :show-inheritance:
```
