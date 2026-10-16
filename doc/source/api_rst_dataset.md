(api-rst-dataset)=
# Dataset.rst accessor

The {class}`Dataset.rst <geoutils.DatasetRasterAccessor>` accessor exposes GeoUtils raster methods on {class}`xarray.Dataset` objects.

It applies operations through {class}`DataArray.rst <geoutils.DataArrayRasterAccessor>` to all compatible raster variables
by default, or to those selected with `variables`.

**For the main raster API reference page, see {ref}`raster-api`.**
Below is the full class description specific to {class}`Dataset.rst <geoutils.DatasetRasterAccessor>`:

```{eval-rst}
.. currentmodule:: geoutils

.. autoclass:: DatasetRasterAccessor
    :members:
    :undoc-members: false
    :show-inheritance:
```
