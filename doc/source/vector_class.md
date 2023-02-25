(vector-class)=

# The georeferenced vector ({class}`~geoutils.Vector`)

Below, a summary of the {class}`~geoutils.Vector` object and its methods.

## Object definition and attributes

A {class}`~geoutils.Vector` contains **a single main attribute**: a {class}`~geopandas.GeoDataFrame` as {attr}`~geoutils.Vector.ds`.

All other attributes are derivatives of the {class}`~geopandas.GeoDataFrame`.

```{important}
In short, {class}`~geoutils.Vector` is a "convenience" composition class built on top of GeoPandas, to facilitate interfacing with {class}`~geoutils.Raster`,
and to allow the addition of more complex vector functionalities.

**All of {class}`~geopandas.GeoDataFrame`'s methods are available directly from a {class}`~geoutils.Vector`**, as if working directly on the {class}`~geopandas.GeoDataFrame`.
```

## Open and save


## Arithmetic


## Reproject


## Crop


## Rasterize

The {func}`~geoutils.Vector.rasterize` operation to convert from {class}`~geoutils.Vector` to {class}`~geoutils.Raster` inherently requires a
{attr}`~geoutils.Raster.res` or {attr}`~geoutils.Raster.shape` attribute to define the grid. While those can be passed on their own.

## Proximity


## Create a `Mask`


## Buffering
