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
(core-composition)=

# Composition from Rasterio and GeoPandas

GeoUtils' main classes {class}`~geoutils.Raster` and {class}`~geoutils.Vector` are linked to [Rasterio](https://rasterio.readthedocs.io/en/latest/) and
[GeoPandas](https://geopandas.org/en/stable/docs.html), respectively, through [class composition](https://realpython.com/inheritance-composition-python/#whats-composition).

They directly rely on their robust geospatial handling functionalities, as well of that of [PyProj](https://pyproj4.github.io/pyproj/stable/index.html), and
add a layer on top for interfacing between rasters and vectors with higher-level operations, performing easier numerical analysis, and adding more advanced geospatial functionalities.

## The {class}`~geoutils.Raster` class composition

The {class}`~geoutils.Raster` is a composition class with **four main attributes**:

1. a {class}`numpy.ma.MaskedArray` as {attr}`~geoutils.Raster.data`,
2. an [{class}`affine.Affine`](https://rasterio.readthedocs.io/en/stable/topics/migrating-to-v1.html#affine-affine-vs-gdal-style-geotransforms) as {attr}`~geoutils.Raster.transform`,
3. a {class}`pyproj.crs.CRS` as {attr}`~geoutils.Raster.crs`, and
4. a {class}`float` or {class}`int` as {attr}`~geoutils.Raster.nodata`.

```{code-cell} ipython3
:tags: [hide-output]

import geoutils as gu

# Instantiate a raster from a filename on disk
filename_rast = gu.examples.get_path("exploradores_aster_dem")
rast = gu.Raster(filename_rast)
rast
```

From these **four main attributes**, many other derivatives attributes exist, such as {attr}`~geoutils.Raster.bounds` or {attr}`~geoutils.Raster.res` to
describe georeferencing. When a {class}`~geoutils.Raster` is based on an **on-disk** dataset, other attributes exist such as {attr}`~geoutils.Raster.name` or
{attr}`~geoutils.Raster.driver`, see {ref}`raster-class` for a summary, or the {ref}`dedicated sections of the API<api-raster-attrs>` for a full listing.

```{note}
By default, {attr}`~geoutils.Raster.data` is not loaded during instantiation. See {ref}`core-lazy-load` for more details.
```

```{code-cell} ipython3
:tags: [hide-output]
# Show summarized information
rast.info()
```

```{important}
The {class}`~geoutils.Raster` is not a composition of either a {class}`rasterio.io.DatasetReader`, a {class}`rasterio.io.MemoryFile` or a {class}`rasterio.io.DatasetWriter`.
It is only linked to those objects to initiate a {class}`~geoutils.Raster` instance which first loads the metadata (notably the three main metadata attributes
{attr}`~geoutils.Raster.crs`, {attr}`~geoutils.Raster.transform` and {attr}`~geoutils.Raster.nodata`).
Then, explicitly or implicitly, it can also {class}`~geoutils.Raster.load` the array data when an **on-disk** dataset exists, or {class}`~geoutils.Raster.to_file` the **in-memory**
dataset to a file.
```

A {class}`~geoutils.Raster` generally exists only as an **in-memory** dataset, not linked to anything else than a {class}`numpy.ma.MaskedArray` in {attr}`~geoutils.Raster.data`
(for instance, when creating from {func}`~geoutils.Raster.from_array`). If a numerical operation is performed on a {class}`~geoutils.Raster` opened from an
**on-disk** dataset, the new dataset will be modified **in-memory**.

```{note}
To check if a {class}`~geoutils.Raster` was loaded from the **on-disk** data, use the {attr}`~geoutils.Raster.is_loaded` attribute.

To check if the {class}`~geoutils.Raster` was modified since loading from the **on-disk** data, use the {attr}`~geoutils.Raster.is_modified` attribute.
```

See {ref}`raster-class` for more details.


## The {class}`~geoutils.Vector` class composition

A {class}`~geoutils.Vector` is a composition class with a single main attribute: a {class}`~geopandas.GeoDataFrame` as {attr}`~geoutils.Vector.ds`.

A {class}`~geoutils.Vector`'s dataframe {attr}`~geoutils.Vector.ds` is directly loaded in-memory
(might evolve towards lazy behaviour soon through [Dask-GeoPandas](https://dask-geopandas.readthedocs.io/en/stable/)).

```{code-cell} ipython3
:tags: [hide-output]
# Instantiate a vector from a filename on disk
filename_vect = gu.examples.get_path("exploradores_rgi_outlines")
vect = gu.Vector(filename_vect)
vect
```

```{code-cell} ipython3
:tags: [hide-output]
# Show summarized information
vect.info()
```

All geospatial methods of {class}`~geopandas.GeoDataFrame` are directly available into {class}`~geoutils.Vector`, and cast the output logically depending on
its type: to a {class}`~geoutils.Vector` for a geometric output (e.g., {class}`~geoutils.Vector.boundary`), or to {class}`pandas.Series` that can be immediately appended to the
{class}`~geoutils.Vector` for a per-feature non-geometric output (e.g., {class}`~geoutils.Vector.area`).

```{code-cell} ipython3
:tags: [hide-output]
# Compute the vector's boundary
vect.boundary
```

See {ref}`vector-class` for more details.
