---
file_format: mystnb
kernelspec:
  name: geoutils
---
(raster-class)=

# The georeferenced raster ({class}`~geoutils.Raster`)

Below, a summary of the {class}`~geoutils.Raster` object and its methods.

(raster-obj-def)=

## Object definition and attributes

A {class}`~geoutils.Raster` contains **four main attributes**:

1. a {class}`numpy.ma.MaskedArray` as {attr}`~geoutils.Raster.data`,
2. an {class}`affine.Affine` as {attr}`~geoutils.Raster.transform`
3. a {class}`pyproj.crs.CRS` as {attr}`~geoutils.Raster.crs`, and
4. a {class}`float` or {class}`int` as {attr}`~geoutils.Raster.nodata`.

For more details on {class}`~geoutils.Raster` class composition, see {ref}`core-composition`.

A {class}`~geoutils.Raster` also contains many derivative attributes, with naming generally consistent with that of a {class}`rasterio.DatasetReader`.

A first category includes georeferencing attributes directly derived from {attr}`~geoutils.Raster.transform`, namely: {attr}`~geoutils.Raster.shape`, 
{attr}`~geoutils.Raster.height`, {attr}`~geoutils.Raster.width`, {attr}`~geoutils.Raster.res`, {attr}`~geoutils.Raster.bounds`.

A second category concerns the attributes derived from the raster array shape and type: {attr}`~geoutils.Raster.count`, {attr}`~geoutils.Raster.indexes` and 
{attr}`~geoutils.Raster.dtypes`. The two former refer to the number of bands loaded in a {class}`~geoutils.Raster`, and their indexes.

```{important}
The {attr}`~geoutils.Raster.indexes` of {class}`rasterio.DatasetReader` start from 1 and not 0, be careful when instantiating or loading from a 
multi-band raster!
```

Finally, the remaining attributes are only relevant when instantiating from a **on-disk** file: {attr}`~geoutils.Raster.name`, {attr}`~geoutils.Raster.driver`, 
{attr}`~geoutils.Raster.count_on_disk`, {attr}`~geoutils.Raster.indexes_on_disk`, {attr}`~geoutils.Raster.is_loaded` and {attr}`~geoutils.Raster.is_modified`.


```{note}
The {attr}`~geoutils.Raster.count` and {attr}`~geoutils.Raster.indexes` attributes always exist, while {attr}`~geoutils.Raster.count_on_disk` and 
{attr}`~geoutils.Raster.indexes_on_disk` only refers to the number of bands on the **on-disk** dataset, if it exists.

For example, {attr}`~geoutils.Raster.count` and {attr}`~geoutils.Raster.count_on_disk` will differ when a single band is loaded from a 
3-band **on-disk** file, by passing a single index to the `indexes` argument in {class}`~geoutils.Raster` or {func}`~geoutils.Raster.load`.
```

The complete list of {class}`~geoutils.Raster` attributes with description is available in {ref}`dedicated sections of the API<api-raster-attrs>`.

## Open and save

A {class}`~geoutils.Raster` is opened by instantiating with either a {class}`str`, a {class}`pathlib.Path` or a {class}`rasterio.DatasetReader`.


```{code-cell} ipython3
import geoutils as gu

# Initiate a raster from disk
raster = gu.Raster(gu.examples.get_path("exploradores_aster_dem"))
raster
```

Detailed information on the {class}`~geoutils.Raster` is printed using {class}`~geoutils.Raster.info()`, along with basic statistics using `stats=True`:

```{code-cell} ipython3
# Print details of raster
print(raster.info(stats=True))
```


```{note}
Calling {class}`~geoutils.Raster.info()` with `stats=True` automatically loads the array in-memory, like any other operation calling {attr}`~geoutils.Raster.data`.
```

A {class}`~geoutils.Raster` is saved to file by calling {func}`~geoutils.Raster.save` with a {class}`str` or a {class}`pathlib.Path`.

```{code-cell} ipython3
# Save raster to disk
raster.save("myraster.tif")
```
```{code-cell} ipython3
:tags: [remove-cell]
import os
os.remove("myraster.tif")
```

## Create from array

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

# Create a Raster from array
raster = gu.Raster.from_array(
        data = ma,
        transform = rio.transform.from_bounds(0, 0, 1, 1, 3, 3),
        crs = pyproj.CRS.from_epsg(4326),
        nodata = 255
    )
raster
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
raster.data
```

For those less familiar with {class}`MaskedArrays<numpy.ma.MaskedArray>` and the associated functions in NumPy, an unmasked {class}`~numpy.ndarray` filled with 
{class}`~numpy.nan` on masked values can be extracted using {func}`~geoutils.Raster.get_nanarray`.

```{code-cell} ipython3
# Get raster's nan-array
raster.get_nanarray()
```

```{important}
Getting a {class}`~numpy.ndarray` filled with {class}`~numpy.nan` will automatically cast the {class}`dtype<numpy.dtype>` to {class}`numpy.float32`. This 
might result in larger memory usage than in the original {class}`~geoutils.Raster` (if of {class}`int` type).

Thanks to the {ref}`core-array-funcs`, **NumPy functions applied directly to a {class}`~geoutils.Raster` will respect {class}`~geoutils.Raster.nodata` 
values** as well as if computing with the {class}`~numpy.ma.MaskedArray` or an unmasked {class}`~numpy.ndarray` filled with {class}`~numpy.nan`.

Additionally, the {class}`~geoutils.Raster` will automatically cast between different {class}`dtypes<numpy.dtype>`, and possibly re-define missing 
{class}`nodatas<geoutils.Raster.nodata>`.
```

## Arithmetic

A {class}`~geoutils.Raster` can be applied any pythonic arithmetic operation ({func}`+<operator.add>`, {func}`-<operator.sub>`, {func}`/<operator.truediv>`, {func}`//<operator.floordiv>`, {func}`*<operator.mul>`, 
{func}`**<operator.pow>`, {func}`%<operator.mod>`) with another {class}`~geoutils.Raster`, {class}`~numpy.ndarray` or number. It will output one or two {class}`Rasters<geoutils.Raster>`. NumPy coercion rules apply for {class}`dtypes<numpy.dtype>`.

```{code-cell} ipython3
# Add 1 and divide raster by 2
(raster + 1)/2
```

A {class}`~geoutils.Raster` can also be applied any pythonic logical comparison operation ({func}`==<operator.eq>`, {func}` != <operator.ne>`, {func}`>=<operator.ge>`, {func}`><operator.gt>`, {func}`<=<operator.le>`, 
{func}`<<operator.lt>`) with another {class}`~geoutils.Raster`, {class}`~numpy.ndarray` or number. It will cast to a {class}`~geoutils.Mask`.

```{code-cell} ipython3
# What raster pixels are less than 100?
raster < 100
```

See {ref}`core-py-ops` for more details.

## Array interface

A {class}`~geoutils.Raster` can be applied any NumPy universal functions and most mathematical, logical or masked-array functions with another 
{class}`~geoutils.Raster`, {class}`~numpy.ndarray` or number.

```{code-cell} ipython3
# Compute the element-wise sqrt
np.sqrt(raster)
```

Logical comparison functions will cast to a {class}`~geoutils.Mask`.

```{code-cell} ipython3
# Is the raster close to another within tolerance?
np.isclose(raster, raster+0.05, atol=0.1)
```

See {ref}`core-array-funcs` for more details.

## Reproject

Reprojecting a {class}`~geoutils.Raster` means to enforce a new {attr}`~geoutils.Raster.transform` and/or {class}`~geoutils.Raster.crs`. This is done by the 
{func}`~geoutils.Raster.reproject` function.

```{important}
As with all geospatial handling methods, the {func}`~geoutils.Raster.reproject` function can be passed only a {class}`~geoutils.Raster` or {class}`~geoutils.
Vector` as argument. 

A {class}`~geoutils.Raster` reference will enforce to match its {attr}`~geoutils.Raster.transform` and {class}`~geoutils.Raster.crs`.
A {class}`~geoutils.Vector` reference will enforce to match its {attr}`~geoutils.Vector.bounds` and {class}`~geoutils.Vector.crs`.

See {ref}`core-match-ref` for more details.
```

The {func}`~geoutils.Raster.reproject` function can also be passed any individual arguments such as `dst_bounds`, to enforce specific georeferencing 
attributes. For more details, see the {ref}`specific section and function descriptions in the API<api-geo-handle>`.

```{code-cell} ipython3
# Original bounds and resolution
print(raster.res)
print(raster.bounds)
```

```{code-cell} ipython3
# Reproject to smaller bounds and higher resolution
raster = raster.reproject(
    dst_res=0.25, 
    dst_bounds={"left": 0, "bottom": 0, "right": 0.75, "top": 0.75}, 
    resampling="cubic")
raster
```

```{code-cell} ipython3
# New bounds and resolution
print(raster.res)
print(raster.bounds)
```

```{note}
In GeoUtils, `"bilinear"` is the default resampling method. A simple {class}`str` matching the naming of a {class}`rasterio.enums.Resampling` method can be 
passed.
```


## Crop

{func}`geoutils.Raster.reproject`

For rasters with different coordinate systems, resolutions or grids, reprojection is needed to fit one raster to another.
`Raster.reproject()` is apt for these use-cases:

```{literalinclude} code/raster-basics_cropping_and_reprojecting.py
:lines: 11
```

This call will crop, project and resample the `larger_raster` to fit the `smaller_raster` exactly.
By default, `Raster.resample()` uses nearest neighbour resampling, which is good for fast reprojections, but may induce unintended artefacts when precision is important.
It is therefore recommended to choose the method that fits the purpose best, using the `resampling=` keyword argument:

1. `resampling="nearest"`: Default. Performant but is not good for changes in resolution and grid.
2. `resampling="bilinear"`: Good when changes in resolution and grid are involved.
3. `resampling="cubic_spline"`: Often considered the best approach. Not as performant as simpler methods.

All valid resampling methods can be seen in the [Rasterio documentation](https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling).

```{eval-rst}
.. minigallery:: geoutils.Raster
        :add-heading:
        :heading-level: -
```

## Polygonize

## Proximity

## Interpolate or extract to point

## Export