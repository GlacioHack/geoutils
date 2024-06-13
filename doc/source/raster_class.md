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
(raster-class)=

# The georeferenced raster ({class}`~geoutils.Raster`)

Below, a summary of the {class}`~geoutils.Raster` object and its methods.

(raster-obj-def)=

## Object definition and attributes

A {class}`~geoutils.Raster` contains **four main attributes**:

1. a {class}`numpy.ma.MaskedArray` as {attr}`~geoutils.Raster.data`, of either {class}`~numpy.integer` or {class}`~numpy.floating` {class}`~numpy.dtype`,
2. an [{class}`affine.Affine`](https://rasterio.readthedocs.io/en/stable/topics/migrating-to-v1.html#affine-affine-vs-gdal-style-geotransforms) as {attr}`~geoutils.Raster.transform`,
3. a {class}`pyproj.crs.CRS` as {attr}`~geoutils.Raster.crs`, and
4. a {class}`float` or {class}`int` as {attr}`~geoutils.Raster.nodata`.

For more details on {class}`~geoutils.Raster` class composition, see {ref}`core-composition`.

A {class}`~geoutils.Raster` also contains many derivative attributes, with naming generally consistent with that of a {class}`rasterio.io.DatasetReader`.

A first category includes georeferencing attributes directly derived from {attr}`~geoutils.Raster.transform`, namely: {attr}`~geoutils.Raster.shape`,
{attr}`~geoutils.Raster.height`, {attr}`~geoutils.Raster.width`, {attr}`~geoutils.Raster.res`, {attr}`~geoutils.Raster.bounds`.

A second category concerns the attributes derived from the raster array shape and type: {attr}`~geoutils.Raster.count`, {attr}`~geoutils.Raster.bands` and
{attr}`~geoutils.Raster.dtype`. The two former refer to the number of bands loaded in a {class}`~geoutils.Raster`, and the band indexes.

```{important}
The {attr}`~geoutils.Raster.bands` of {class}`rasterio.io.DatasetReader` start from 1 and not 0, be careful when instantiating or loading from a
multi-band raster!
```

Finally, the remaining attributes are only relevant when instantiating from a **on-disk** file: {attr}`~geoutils.Raster.name`, {attr}`~geoutils.Raster.driver`,
{attr}`~geoutils.Raster.count_on_disk`, {attr}`~geoutils.Raster.bands_on_disk`, {attr}`~geoutils.Raster.is_loaded` and {attr}`~geoutils.Raster.is_modified`.


```{note}
The {attr}`~geoutils.Raster.count` and {attr}`~geoutils.Raster.bands` attributes always exist, while {attr}`~geoutils.Raster.count_on_disk` and
{attr}`~geoutils.Raster.bands_on_disk` only refers to the number of bands on the **on-disk** dataset, if it exists.

For example, {attr}`~geoutils.Raster.count` and {attr}`~geoutils.Raster.count_on_disk` will differ when a single band is loaded from a
3-band **on-disk** file, by passing a single index to the `bands` argument in {class}`~geoutils.Raster` or {func}`~geoutils.Raster.load`.
```

The complete list of {class}`~geoutils.Raster` attributes with description is available in {ref}`dedicated sections of the API<api-raster-attrs>`.

## Open and save

A {class}`~geoutils.Raster` is opened by instantiating with either a {class}`str`, a {class}`pathlib.Path`, a {class}`rasterio.io.DatasetReader` or a
{class}`rasterio.io.MemoryFile`.


```{code-cell} ipython3
import geoutils as gu

# Instantiate a raster from a filename on disk
filename_rast = gu.examples.get_path("exploradores_aster_dem")
rast = gu.Raster(filename_rast)
rast
```

Detailed information on the {class}`~geoutils.Raster` is printed using {func}`~geoutils.Raster.info`, along with basic statistics using `stats=True`:

```{code-cell} ipython3
# Print details of raster
print(rast.info(stats=True))
```

```{note}
Calling {class}`~geoutils.Raster.info()` with `stats=True` automatically loads the array in-memory, like any other operation calling {attr}`~geoutils.Raster.data`.
```

A {class}`~geoutils.Raster` is saved to file by calling {func}`~geoutils.Raster.save` with a {class}`str` or a {class}`pathlib.Path`.

```{code-cell} ipython3
# Save raster to disk
rast.save("myraster.tif")
```
```{code-cell} ipython3
:tags: [remove-cell]
import os
os.remove("myraster.tif")
```

## Create from {class}`~numpy.ndarray`

A {class}`~geoutils.Raster` is created from an array by calling the class method {func}`~geoutils.Raster.from_array` and passing the
{ref}`four main attributes<raster-obj-def>`.

```{code-cell} ipython3
import rasterio as rio
import pyproj
import numpy as np

# Create a random 3 x 3 masked array
np.random.seed(42)
arr = np.random.randint(0, 255, size=(3, 3), dtype="uint8")
mask = np.random.randint(0, 2, size=(3, 3), dtype="bool")
ma = np.ma.masked_array(data=arr, mask=mask)

# Create a raster from array
rast = gu.Raster.from_array(
       data = ma,
       transform = rio.transform.from_bounds(0, 0, 1, 1, 3, 3),
       crs = pyproj.CRS.from_epsg(4326),
       nodata = 255
    )
rast
```

```{important}
The {attr}`~geoutils.Raster.data` attribute can be passed as an unmasked {class}`~numpy.ndarray`. That array will be converted to a {class}`~numpy.ma.MaskedArray`
with a {class}`~numpy.ma.MaskedArray.mask` on all {class}`~numpy.nan` and {class}`~numpy.inf` in the {attr}`~geoutils.Raster.data`, and on all values
matching the {attr}`~geoutils.Raster.nodata` value passed to {func}`~geoutils.Raster.from_array`.
```

## Get array

The array of a {class}`~geoutils.Raster` is available in {class}`~geoutils.Raster.data` as a {class}`~numpy.ma.MaskedArray`.

```{code-cell} ipython3
# Get raster's masked-array
rast.data
```

For those less familiar with {class}`MaskedArrays<numpy.ma.MaskedArray>` and the associated functions in NumPy, an unmasked {class}`~numpy.ndarray` filled with
{class}`~numpy.nan` on masked values can be extracted using {func}`~geoutils.Raster.get_nanarray`.

```{code-cell} ipython3
# Get raster's nan-array
rast.get_nanarray()
```

```{important}
Getting a {class}`~numpy.ndarray` filled with {class}`~numpy.nan` will automatically cast the {class}`dtype<numpy.dtype>` to {class}`numpy.float32`. This
might result in larger memory usage than in the original {class}`~geoutils.Raster` (if of {class}`int` type).

Thanks to the {ref}`core-array-funcs`, **NumPy functions applied directly to a {class}`~geoutils.Raster` will respect {class}`~geoutils.Raster.nodata`
values** as well as if computing with the {class}`~numpy.ma.MaskedArray` or an unmasked {class}`~numpy.ndarray` filled with {class}`~numpy.nan`.

Additionally, the {class}`~geoutils.Raster` will automatically cast between different {class}`dtype<numpy.dtype>`, and possibly re-define missing
{class}`nodatas<geoutils.Raster.nodata>`.
```

## Arithmetic

A {class}`~geoutils.Raster` can be applied any pythonic arithmetic operation ({func}`+<operator.add>`, {func}`-<operator.sub>`, {func}`/<operator.truediv>`, {func}`//<operator.floordiv>`, {func}`*<operator.mul>`,
{func}`**<operator.pow>`, {func}`%<operator.mod>`) with another {class}`~geoutils.Raster`, {class}`~numpy.ndarray` or number. It will output one or two {class}`Rasters<geoutils.Raster>`. NumPy coercion rules apply for {class}`dtype<numpy.dtype>`.

```{code-cell} ipython3
# Add 1 and divide raster by 2
(rast + 1)/2
```

A {class}`~geoutils.Raster` can also be applied any pythonic logical comparison operation ({func}`==<operator.eq>`, {func}` != <operator.ne>`, {func}`>=<operator.ge>`, {func}`><operator.gt>`, {func}`<=<operator.le>`,
{func}`<<operator.lt>`) with another {class}`~geoutils.Raster`, {class}`~numpy.ndarray` or number. It will cast to a {class}`~geoutils.Mask`.

```{code-cell} ipython3
# What raster pixels are less than 100?
rast < 100
```

See {ref}`core-py-ops` for more details.

## Array interface

A {class}`~geoutils.Raster` can be applied any NumPy universal functions and most mathematical, logical or masked-array functions with another
{class}`~geoutils.Raster`, {class}`~numpy.ndarray` or number.

```{code-cell} ipython3
# Compute the element-wise square-root
np.sqrt(rast)
```

Logical comparison functions will cast to a {class}`~geoutils.Mask`.

```{code-cell} ipython3
# Is the raster close to another within tolerance?

np.isclose(rast, rast+0.05, atol=0.1)
```

See {ref}`core-array-funcs` for more details.

## Reproject

Reprojecting a {class}`~geoutils.Raster` is done through the {func}`~geoutils.Raster.reproject` function, which enforces new {attr}`~geoutils.Raster.transform`
and/or
{class}`~geoutils.Raster.crs`.

```{important}
As with all geospatial handling methods, the {func}`~geoutils.Raster.reproject` function can be passed a {class}`~geoutils.Raster` or
{class}`~geoutils.Vector` as a reference to match. In that case, no other argument is necessary.

A {class}`~geoutils.Raster` reference will enforce to match its {attr}`~geoutils.Raster.transform` and {class}`~geoutils.Raster.crs`.
A {class}`~geoutils.Vector` reference will enforce to match its {attr}`~geoutils.Vector.bounds` and {class}`~geoutils.Vector.crs`.

See {ref}`core-match-ref` for more details.
```

The {func}`~geoutils.Raster.reproject` function can also be passed any individual arguments such as `dst_bounds`, to enforce specific georeferencing
attributes. For more details, see the {ref}`specific section and function descriptions in the API<api-geo-handle>`.

```{code-cell} ipython3
# Original bounds and resolution
print(rast.res)
print(rast.bounds)
```

```{code-cell} ipython3
# Reproject to smaller bounds and higher resolution
rast_reproj = rast.reproject(
    res=0.1,
    bounds={"left": 0, "bottom": 0, "right": 0.75, "top": 0.75},
    resampling="cubic")
rast_reproj
```

```{code-cell} ipython3
# New bounds and resolution
print(rast_reproj.res)
print(rast_reproj.bounds)
```

```{note}
In GeoUtils, `"bilinear"` is the default resampling method. A simple {class}`str` matching the naming of a {class}`rasterio.enums.Resampling` method can be
passed.

Resampling methods are listed in **[the dedicated section of Rasterio's API](https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling)**.
```

## Crop

Cropping a {class}`~geoutils.Raster` is done through the {func}`~geoutils.Raster.crop` function, which enforces new {attr}`~geoutils.Raster.bounds`.

```{important}
As with all geospatial handling methods, the {func}`~geoutils.Raster.crop` function can be passed only a {class}`~geoutils.Raster` or {class}`~geoutils.Vector`
as a reference to match. In that case, no other argument is necessary.

See {ref}`core-match-ref` for more details.
```

The {func}`~geoutils.Raster.crop` function can also be passed a {class}`list` or {class}`tuple` of bounds (`xmin`, `ymin`, `xmax`, `ymax`). By default,
{func}`~geoutils.Raster.crop` returns a new Raster.
For more details, see the {ref}`specific section and function descriptions in the API<api-geo-handle>`.

```{code-cell} ipython3
# Crop raster to smaller bounds
rast_crop = rast.crop(crop_geom=(0.3, 0.3, 1, 1))
print(rast_crop.bounds)
```

## Polygonize

Polygonizing a {class}`~geoutils.Raster` is done through the {func}`~geoutils.Raster.polygonize` function, which converts target pixels into a multi-polygon
{class}`~geoutils.Vector`.

```{note}
For a {class}`~geoutils.Mask`, {func}`~geoutils.Raster.polygonize` implicitly targets `True` values and thus does not require target pixels. See
{ref}`mask-class-poly-overloaded`.
```

```{code-cell} ipython3
# Polygonize all values lower than 100
vect_lt_100 = (rast < 100).polygonize()
vect_lt_100
```

## Proximity

Computing proximity from a {class}`~geoutils.Raster` is done through by the {func}`~geoutils.Raster.proximity` function, which computes the closest distance
to any target pixels in the {class}`~geoutils.Raster`.

```{note}
For a {class}`~geoutils.Mask`, {func}`~geoutils.Raster.proximity` implicitly targets `True` values and thus does not require target pixels. See
{ref}`mask-class-prox-overloaded`.
```

```{code-cell} ipython3
# Compute proximity from mask for all values lower than 100
prox_lt_100 = (rast < 100).proximity()
prox_lt_100
```

Optionally, instead of target pixel values, a {class}`~geoutils.Vector` can be passed to compute the proximity from the geometry.

```{code-cell} ipython3
# Compute proximity from mask for all values lower than 100
prox_lt_100_from_vect = rast.proximity(vector=vect_lt_100)
prox_lt_100_from_vect
```

## Interpolate or reduce to point

Interpolating or extracting {class}`~geoutils.Raster` values at specific points can be done through:
- the {func}`~geoutils.Raster.reduce_points` function, that applies a reductor function ({func}`numpy.ma.mean` by default) to a surrounding window for each coordinate, or
- the {func}`~geoutils.Raster.interp_points` function, that interpolates the {class}`~geoutils.Raster`'s regular grid to each coordinate using a resampling algorithm.

```{code-cell} ipython3
# Extract median value in a 3 x 3 pixel window
rast_reproj.reduce_points((0.5, 0.5), window=3, reducer_function=np.ma.median)
```

```{code-cell} ipython3
# Interpolate coordinate value with quintic algorithm
rast_reproj.interp_points((0.5, 0.5), method="quintic")
```

```{note}
Both {func}`~geoutils.Raster.reduce_points` and {func}`~geoutils.Raster.interp_points` can be passed a single coordinate as {class}`floats<float>`, or a
{class}`list` of coordinates.
```

## Export

A {class}`~geoutils.Raster` can be exported to different formats, to facilitate inter-compatibility with different packages and code versions.

Those include exporting to:
- a {class}`xarray.Dataset` with {class}`~geoutils.Raster.to_xarray`,
- a {class}`rasterio.io.DatasetReader` with {class}`~geoutils.Raster.to_rio_dataset`,
- a {class}`numpy.ndarray` or {class}`geoutils.Vector` as a point cloud with {class}`~geoutils.Raster.to_pointcloud`.

```{code-cell} ipython3
# Export to rasterio dataset-reader through a memoryfile
rast_reproj.to_rio_dataset()
```

```{code-cell} ipython3
# Export to geopandas dataframe
rast_reproj.to_pointcloud()
```

```{code-cell} ipython3
# Export to xarray data array
rast_reproj.to_xarray()
```
