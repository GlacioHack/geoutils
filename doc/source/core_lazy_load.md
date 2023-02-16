---
file_format: mystnb
kernelspec:
  name: geoutils
---
(core-lazy-load)=

# Implicit lazy loading

## Lazy instantiation of {class}`Rasters<geoutils.Raster>`

By default, GeoUtils instantiate a {class}`~geoutils.Raster` from an **on-disk** file without loading its {attr}`geoutils.Raster.data` array. It only loads its 
metadata ({attr}`~geoutils.Raster.transform`, {attr}`~geoutils.Raster.crs`, {attr}`~geoutils.Raster.nodata` and derivatives, as well as 
{attr}`~geoutils.Raster.name` and {attr}`~geoutils.Raster.driver`).

```{code-cell} ipython3

import geoutils as gu

# Initiate a Raster from disk
raster = gu.Raster(gu.examples.get_path("everest_landsat_b4"))

# This Raster is not loaded
raster
```

To load the data explicity during instantiation opening, `load_data=True` can be passed to {class}`~geoutils.Raster`. Or the {func}`~geoutils.Raster.load` 
method can be called after. The two are equivalent.

```{code-cell} ipython3
# Initiate another Raster just for the purpose of loading
raster_toload = gu.Raster(gu.examples.get_path("everest_landsat_b4"))
raster_toload.load()

# This Raster is loaded
raster_toload
```

## Lazy passing of georeferencing metadata

Operations relying on georeferencing metadata of {class}`Rasters<geoutils.Raster>` or {class}`Vectors<geoutils.Vector>` are always done by respecting the 
possible lazy loading of the objects.

For instance, using any {class}`~geoutils.Raster` or {class}`~geoutils.Vector` as a match-reference for a geospatial operation (see {ref}`core-match-ref`) will 
always conserve the lazy loading of that match-reference object.

```{code-cell} ipython3
# Use a smaller Raster as reference to crop the initial one
smaller_raster = gu.Raster(gu.examples.get_path("everest_landsat_b4_cropped"))
raster.crop(smaller_raster)

# The reference Raster is not loaded
smaller_raster
```

## Optimized geospatial subsetting

```{important}
These features are a work in progress, we aim to make GeoUtils more lazy-friendly through [Dask](https://docs.dask.org/en/stable/) in future versions of the 
package!
```

Some georeferencing operations be done without loading the entire array Right now, relying directly on Rasterio, GeoUtils supports optimized subsetting 
through the {func}`~geoutils.Raster.`crop` method.

```{code-cell} ipython3
# The previously cropped Raster was loaded without accessing the entire array
raster
```



