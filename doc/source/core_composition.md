---
file_format: mystnb
kernelspec:
  name: geoutils
---
(core-composition)=

# Composition from Rasterio and GeoPandas

GeoUtils' main classes {class}`~geoutils.Raster` and {class}`~geoutils.Vector` are linked to [Rasterio](https://rasterio.readthedocs.io/en/latest/) and
[GeoPandas](https://geopandas.org/en/stable/docs.html), respectively, through class composition. 

They directly rely on their robust geospatial handling functionalities, as well of that of [PyProj](https://pyproj4.github.io/pyproj/stable/index.html), and 
add a layer on top for interfacing between rasters and vectors with higher-level operations, performing easier numerical analysis, and adding more advanced geospatial functionalities.

## The {class}`~geoutils.Raster` class composition

The {class}`~geoutils.Raster` is a composition class with **four main attributes**:

1. a {class}`numpy.ma.MaskedArray` as {attr}`~geoutils.Raster.data`,
2. an {class}`affine.Affine` as {attr}`~geoutils.Raster.transform`
3. a {class}`pyproj.crs.CRS` as {attr}`~geoutils.Raster.crs`, and
4. a {class}`float` or {class}`int` as {attr}`~geoutils.Raster.nodata`.

```{code-cell} ipython3
:tags: [hide-output]

import geoutils as gu

# Initiate a Raster from disk
raster = gu.Raster(gu.examples.get_path("exploradores_aster_dem"))
raster
```

From these **four main attributes**, many other derivatives attributes exist, such as {attr}`~geoutils.Raster.bounds` or {attr}`~geoutils.Raster.res` to 
describe georeferencing. When a {class}`~geoutils.Raster` is based on an **on-disk** dataset, other attributes exist such as {attr}`~geoutils.Raster.
name` or {attr}`~geoutils.Raster.driver`, see {ref}`raster-class` for a summary, or the {ref}`dedicated sections of the API` for a full listing.

```{note}
By default, {attr}`~geoutils.Raster.data` is not loaded during instantiation. See {ref}`core-lazy-load` for more details.
```

```{code-cell} ipython3
:tags: [hide-output]
# Show summarized information
print(raster.info())
```

```{important}
The {class}`~geoutils.Raster` is not a composition of either a {class}`rasterio.io.DatasetReader`, a {class}`rasterio.io.MemoryFile` or a {class}`rasterio.io.DatasetWriter`. 
It is only linked to those objects to initiate a {class}`~geoutils.Raster` instance which first loads the metadata (notably the three main metadata attributes 
{attr}`~geoutils.Raster.crs`, {attr}`~geoutils.Raster.transform` and {attr}`~geoutils.Raster.nodata`). 
Then, explicity or implicitly, it can also {class}`~geoutils.Raster.load` the array data when an **on-disk** dataset exists, or {class}`~geoutils.Raster.save` the **in-memory** 
dataset to a file.
```

A {class}`~geoutils.Raster` generally exists only as an **in-memory** dataset, not linked to anything else than a {class}`numpy.ma.MaskedArray` in {attr}`~geoutils.Raster.data` 
(for instance, when creating from {func}`~geoutils.Raster.from_array`). If a numerical operation is performed on a {class}`~geoutils.Raster` opened from an 
**on-disk** dataset, the new dataset will be modified **in-memory**.

```{note}
To check if a {class}`~geoutils.Raster` was loaded from the **on-disk** data, use the {attr}`~geoutils.Raster.is_loaded` attribute. 

To check if the {class}`~geoutils.Raster` was modified since loading from the **on-disk** data, use the {attr}`~geoutils.Raster.is_modified` attribute. 
```

## The {class}`~geoutils.Vector` class composition

A {class}`~geoutils.Vector` is a composition class with a single main attribute: a {class}`~geopandas.GeoDataFrame` as {attr}`~geoutils.Vector.ds`.

Because lazy loading is a lesser priority with vector data, a {class}`~geoutils.Vector` directly loads its {attr}`~geoutils.Vector.ds`. Besides, many 
higher-level geospatial methods are already available in {class}`~geopandas.GeoDataFrame`. We thus only wrap those directly into {class}`~geoutils.Vector`, 
in order to easily call them from the vector object, and build additional methods on top.

```{code-cell} ipython3
:tags: [hide-output]

# Initiate a Vector from disk
vector = examples.get_path("exploradores_rgi_outlines")
vector
```
