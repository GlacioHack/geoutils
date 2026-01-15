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

# The georeferenced vector ({class}`~geoutils.Vector`)

Below, a summary of the {class}`~geoutils.Vector` object and its methods.

## Object definition and attributes

A {class}`~geoutils.Vector` contains **a single main attribute**: a {class}`~geopandas.GeoDataFrame` as {attr}`~geoutils.Vector.ds`.

All other attributes are derivatives of the {class}`~geopandas.GeoDataFrame`.

In short, {class}`~geoutils.Vector` is a "convenience" composition class built on top of GeoPandas, to consistently cast geometric outputs to a
{class}`geoutils.Vector`, facilitate the interface with {class}`~geoutils.Raster`, and allow the addition of more complex vector functionalities.

**All geometric functionalities of {class}`~geopandas.GeoDataFrame`'s methods are available directly from a {class}`~geoutils.Vector`**, as if working
directly on the {class}`~geopandas.GeoDataFrame`. Dataframe functionalities from Pandas can be called from its {attr}`~geoutils.Vector.ds`.

```{caution}
The {attr}`~geoutils.Vector.bounds` attribute of a {class}`~geoutils.Vector` corresponds to the {attr}`~geopandas.GeoDataFrame.total_bounds` attribute of a
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

## Reproject

Reprojecting a {class}`~geoutils.Vector` is done through the {func}`~geoutils.Vector.reproject` function, which enforces a new {attr}`~geoutils.Vector.crs`.

```{important}
As with all geospatial handling methods, the {func}`~geoutils.Vector.reproject` function can be passed a
{class}`~geoutils.Raster` or {class}`~geoutils.Vector` as a reference to match its {class}`~geoutils.Raster.crs`.
In that case, no other argument is necessary.

See {ref}`core-match-ref` for more details.
```

The {func}`~geoutils.Vector.reproject` function can also be passed a `dst_crs` argument directly.

```{code-cell} ipython3
# Original CRS
print(vect.crs)
```

```{code-cell} ipython3
# Open a raster for which we want to match the CRS
filename_rast = gu.examples.get_path("exploradores_aster_dem")
rast = gu.Raster(filename_rast)
# Reproject the vector to the raster's CRS
vect_reproj = vect.reproject(rast)
# New CRS
print(vect_reproj.crs)
```

```{note}
Calling {func}`geoutils.Vector.to_crs` is also possible, but mirrors GeoPandas' API (a match-reference argument cannot be passed).
```

## Crop

Cropping a {class}`~geoutils.Vector` is done through the {func}`~geoutils.Vector.crop` function, which enforces new {attr}`~geoutils.Vector.bounds`.


```{important}
As with all geospatial handling methods, the {func}`~geoutils.Vector.crop` function can be passed a {class}`~geoutils.Raster` or
{class}`~geoutils.Vector` as a reference to match. In that case, no other argument is necessary.

See {ref}`core-match-ref` for more details.
```

The {func}`~geoutils.Vector.crop` function can also be passed a {class}`list` or {class}`tuple` of bounds (`xmin`, `ymin`, `xmax`, `ymax`).

By default, {func}`~geoutils.Vector.crop` returns a new {class}`~geoutils.Vector` which keeps all intersecting geometries. It can also be passed the `clip` argument to clip
intersecting geometries to the extent.

```{code-cell} ipython3
# Crop vector to smaller bounds
vect_crop = vect.crop(crop_geom=(-73.5, -46.6, -73.4, -46.5), clip=True)
vect_crop.info()
```

## Rasterize

Rasterizing a {class}`~geoutils.Vector` to a {class}`~geoutils.Raster` is done through the {func}`~geoutils.Vector.rasterize` function, which converts vector
geometries into gridded values.

By default, the value of index of the {class}`~geoutils.Vector`'s {attr}`~geoutils.Vector.ds` is burned on a raster grid for each respective geometry.

```{note}
If an `out_value` of `0` (default) and `in_value` value of `1` are passed (i.e., boolean output), {func}`~geoutils.Vector.rasterize` will automatically cast
the output to a raster mask, i.e a boolean {class}`~geoutils.Raster`.
```

To define the grid on which to rasterize, a reference {class}`~geoutils.Raster` to match can be passed. Alternatively, a {attr}`~geoutils.Raster.res` or
{attr}`~geoutils.Raster.shape` can be passed to define the grid.

```{code-cell} ipython3
# Rasterize all geometries by index
rasterized_vect = vect.rasterize(rast)
rasterized_vect
```

## Create a raster mask

Creating a raster mask, i.e. a boolean {class}`~geoutils.Raster`, from a {class}`~geoutils.Vector` is done through the {func}`~geoutils.Vector.create_mask`
function, which converts vector geometries into boolean gridded values for all features.

Similarly as for {func}`~geoutils.Vector.rasterize`, the function expects parameters to define the grid on which to rasterize the output. A reference
{class}`~geoutils.Raster` to match can be passed or, alternatively, individual parameters.

```{code-cell} ipython3
# Create a mask of all geometries on the raster grid
mask_vect = vect.create_mask(rast)
mask_vect
```

## Proximity

Computing proximity from a {class}`~geoutils.Vector` is done through by the {func}`~geoutils.Vector.proximity` function, which computes the closest distance
to any geometry in the {class}`~geoutils.Vector`.

Similarly as for {func}`~geoutils.Vector.rasterize`, the function expects parameters to define the grid on which to rasterize the output. A reference
{class}`~geoutils.Raster` to match can be passed or, alternatively, individual parameters.

```{code-cell} ipython3
# Compute proximity from vector on the raster grid
proximity_to_vect = vect.proximity(rast)
proximity_to_vect
```

## Metric buffering

Computing a buffer accurately in a local metric projection is done through the {func}`~geoutils.Vector.buffer_metric` function, which computes the buffer in
a local UTM zone.

```{code-cell} ipython3
:tags: [hide-output]

# Compute buffer of 100 m on the vector
buffered_vect = vect.buffer_metric(100)
buffered_vect
```

Additionally, to prevent buffers from overlapping, the {func}`~geoutils.Vector.buffer_without_overlap` function uses Voronoi polygons to reconcile the buffer
output of each feature.
