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
(vector-class)=

# The georeferenced vector

GeoUtils also exposes vector methods through the {class}`vct <geoutils.VectorAccessor>` accessor on a
{class}`geopandas.GeoDataFrame`. We recommend this interface for new workflows, while the {class}`~geoutils.Vector`
object remains available with the same GeoUtils-specific methods.

Below, a summary of the {class}`~geoutils.Vector` object and its methods.

## Object definition and attributes

A {class}`~geoutils.Vector` contains **a single main attribute**: a {class}`~geopandas.GeoDataFrame` as {attr}`~geoutils.Vector.ds`.

All other attributes are derivatives of the {class}`~geopandas.GeoDataFrame`.

In short, {class}`~geoutils.Vector` is a "convenience" composition class built on top of GeoPandas, to consistently cast geometric outputs to a
{class}`geoutils.Vector`, facilitate the interface with {class}`~geoutils.Raster`, and allow the addition of more complex vector functionalities.

**All geometric functionalities of {class}`~geopandas.GeoDataFrame`'s methods are available directly from a {class}`~geoutils.Vector`**, as if working
directly on the {class}`~geopandas.GeoDataFrame`. Dataframe functionalities from Pandas can be called from its {attr}`~geoutils.Vector.ds`.

```{caution}
The {attr}`~geoutils.Vector.bbox` attribute of a {class}`~geoutils.Vector` corresponds to the {attr}`~geopandas.GeoDataFrame.total_bounds` attribute of a
{class}`~geopandas.GeoDataFrame` converted to a {class}`rasterio.coords.BoundingBox`, for consistency between rasters and vectors.

The equivalent of {attr}`geopandas.GeoDataFrame.bounds` (i.e., a per-feature bounds) for {class}`Vectors<geoutils.Vector>` is {attr}`~geoutils.Vector.geom_bounds`.
```

## Open and save

A {class}`~geoutils.Vector` is opened by instantiating with either a {class}`str`, a {class}`pathlib.Path`, a {class}`geopandas.GeoDataFrame`,
a {class}`geopandas.GeoSeries` or a {class}`shapely.Geometry`.


```{code-cell} ipython3
:tags: [hide-output]

import geoutils as gu

# Instantiate a vector from disk
filename_vect = gu.examples.get_path("exploradores_rgi_outlines")
vect = gu.Vector(filename_vect)
vect
```

Detailed information on the {class}`~geoutils.Vector` is printed using {func}`~geoutils.Vector.info`:

```{code-cell} ipython3
# Print details of vector
vect.info()
```

A {class}`~geoutils.Vector` is saved to file by calling {func}`~geoutils.Raster.to_file` with a {class}`str` or a {class}`pathlib.Path`.

```{code-cell} ipython3
# Save vector to disk
vect.to_file("myvector.gpkg")
```
```{code-cell} ipython3
:tags: [remove-cell]
import os
os.remove("myvector.gpkg")
```

```{note}
GeoPandas functions with the same behaviour such as {func}`geopandas.GeoDataFrame.to_file` can also be used directly on a {class}`~geoutils.Vector`,
for example calling {func}`geoutils.Vector.to_file`.
```


## From Shapely and GeoPandas

Nearly all geometric attributes and functions of GeoPandas (and sometimes, under the hood, Shapely) can be called from a {class}`~geoutils.Vector`.

In {class}`~geoutils.Vector`, those have three types of behaviour:

1. Methods that return a geometric output (e.g., {attr}`~geoutils.Vector.boundary` or {func}`~geoutils.Vector.symmetric_difference`), which are cast into a
   {class}`~geoutils.Vector`,
2. Methods that return a non-geometric series of same length as the number of features (e.g., {attr}`~geoutils.Vector.area` or {func}`~geoutils.Vector.overlaps`),
   which can optionally be appended to the {class}`~geoutils.Vector` (instead of returning of the default {class}`pandas.Series`),
3. Methods that return any other type of output (e.g., {func}`~geoutils.Vector.has_sindex` or {func}`~geoutils.Vector.to_feather`), for which the output is
   preserved.

```{important}
See the full list of supported methods in the {ref}`dedicated section of the API<vector-from-geopandas>`.
```

These behaviours aim to simplify the analysis of vectors, removing the need to operate on many different objects due to varying function outputs
({class}`geopandas.GeoDataFrame`, {class}`geopandas.GeoSeries`, {class}`shapely.Geometry`, {class}`pandas.Series`).

```{code-cell} ipython3
# Example of method with geometric output
vect.boundary
```

```{code-cell} ipython3
---
mystnb:
  output_stderr: show
---

# Example of method with non-geometry output
vect.area
```

```{code-cell} ipython3
# Example of method with other output type
vect.to_json()
```

## Related features

For more details on features applicable to vectors, refer to the following pages!

| Topic | Documentation |
|---|---|
| CRS, bounds, footprints, and local metric projections | {ref}`referencing` |
| Reprojection, crop, clip, rasterization, and masking | {ref}`transformations` |
| Proximity and metric buffering | {ref}`proximity` |
| Match-reference arguments | {ref}`core-match-ref` |
| Lazy and out-of-memory execution | {ref}`scalability-index` |
| Complete GeoUtils method and attribute listing | {ref}`vector-api` |
