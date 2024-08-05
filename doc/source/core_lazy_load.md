---
file_format: mystnb
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: geoutils-env
  language: python
  name: geoutils
---
(core-lazy-load)=

# Implicit lazy loading

Lazy loading, also known as "call-by-need", is the delay in loading or evaluating a dataset.

In GeoUtils, we implicitly load and pass only metadata until the data is actually needed, and are working to implement lazy analysis tools relying on other packages.

## Lazy instantiation of {class}`Rasters<geoutils.Raster>`

By default, GeoUtils instantiate a {class}`~geoutils.Raster` from an **on-disk** file without loading its {attr}`geoutils.Raster.data` array. It only loads its
metadata ({attr}`~geoutils.Raster.transform`, {attr}`~geoutils.Raster.crs`, {attr}`~geoutils.Raster.nodata` and derivatives, as well as
{attr}`~geoutils.Raster.name` and {attr}`~geoutils.Raster.driver`).

```{code-cell} ipython3

import geoutils as gu

# Instantiate a raster from a filename on disk
filename_rast = gu.examples.get_path("everest_landsat_b4")
rast = gu.Raster(filename_rast)

# This raster is not loaded
rast
```

To load the data explicitly during instantiation opening, `load_data=True` can be passed to {class}`~geoutils.Raster`. Or the {func}`~geoutils.Raster.load`
method can be called after. The two are equivalent.

```{code-cell} ipython3
# Initiate another raster just for the purpose of loading
rast_to_load = gu.Raster(gu.examples.get_path("everest_landsat_b4"))
rast_to_load.load()

# This raster is loaded
rast_to_load
```

## Lazy passing of georeferencing metadata

Operations relying on georeferencing metadata of {class}`Rasters<geoutils.Raster>` or {class}`Vectors<geoutils.Vector>` are always done by respecting the
possible lazy loading of the objects.

For instance, using any {class}`~geoutils.Raster` or {class}`~geoutils.Vector` as a match-reference for a geospatial operation (see {ref}`core-match-ref`) will
always conserve the lazy loading of that match-reference object.

```{code-cell} ipython3
---
mystnb:
  output_stderr: show
---

# Use a smaller Raster as reference to crop the initial one
smaller_rast = gu.Raster(gu.examples.get_path("everest_landsat_b4_cropped"))
rast.crop(smaller_rast)

# The reference raster is not loaded
smaller_rast
```

## Optimized geospatial subsetting

```{important}
These features are a work in progress, we aim to make GeoUtils more lazy-friendly through [Dask](https://docs.dask.org/en/stable/) in future versions of the
package!
```

Some georeferencing operations can be done without loading the entire array. Right now, relying directly on Rasterio, GeoUtils supports optimized subsetting
through the {func}`~geoutils.Raster.crop` method.

```{code-cell} ipython3
# The previously cropped Raster was loaded without accessing the entire array
rast
```
