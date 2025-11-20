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

The {attr}`~geoutils.Raster.is_mask` describes if the raster is a mask (i.e. a boolean raster), which overrides the behaviour of some methods to facilitate
their manipulation, as boolean data types are not natively supported by raster filetypes or many operations despite their usefulness for analysis (see
{ref}`mask-type` section for details).


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

A {class}`~geoutils.Raster` is saved to file by calling {func}`~geoutils.Raster.to_file` with a {class}`str` or a {class}`pathlib.Path`.

```{code-cell} ipython3
# Save raster to disk
rast.to_file("myraster.tif")
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
{func}`<<operator.lt>`) with another {class}`~geoutils.Raster`, {class}`~numpy.ndarray` or number. It will cast to a raster mask, i.e. a boolean {class}
`~geoutils.Raster`.

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

Logical comparison functions will cast to a raster mask, i.e. a boolean {class}`~geoutils.Raster` (True or False).

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

[//]: # (```{note})

[//]: # (Reprojecting a {class}`~geoutils.Raster` can be done out-of-memory in multiprocessing by passing a)

[//]: # ({class}`~geoutils.raster.MultiprocConfig` parameter to the {func}`~geoutils.Raster.reproject` function.)

[//]: # (In this case, the reprojected raster is saved on disk under the specify path in {class}`~geoutils.raster.MultiprocConfig` &#40;or a temporary file&#41; and the raster metadata are loaded from the file.)

[//]: # (```)

[//]: # ()
[//]: # (```{code-cell} ipython3)

[//]: # (# Same example out-of-memory)

[//]: # (from geoutils.raster import MultiprocConfig, ClusterGenerator)

[//]: # (cluster = ClusterGenerator&#40;"multi", nb_workers=4&#41;)

[//]: # (mp_config = MultiprocConfig&#40;chunk_size=200, cluster=None&#41;  # Pass a cluster to perform reprojection in multiprocessing)

[//]: # (rast_reproj = rast.reproject&#40;)

[//]: # (    res=0.1,)

[//]: # (    bounds={"left": 0, "bottom": 0, "right": 0.75, "top": 0.75},)

[//]: # (    resampling="cubic",)

[//]: # (    multiproc_config=mp_config&#41;)

[//]: # (rast_reproj)

[//]: # (```)

## Crop

Cropping a {class}`~geoutils.Raster` is done through the {func}`~geoutils.Raster.crop` function, which enforces new {attr}`~geoutils.Raster.bounds`.
Additionally, you can use the {func}`~geoutils.Raster.icrop` method to crop the raster using pixel coordinates instead of geographic bounds.
Both cropping methods can be used before loading the raster's data into memory. This optimization can prevent loading unnecessary parts of the data, which is particularly useful when working with large rasters.

```{important}
As with all geospatial handling methods, the {func}`~geoutils.Raster.crop` function can be passed only a {class}`~geoutils.Raster` or {class}`~geoutils.Vector`
as a reference to match. In that case, no other argument is necessary.

See {ref}`core-match-ref` for more details.
```

The {func}`~geoutils.Raster.crop` function can also be passed a {class}`list` or {class}`tuple` of bounds (`xmin`, `ymin`, `xmax`, `ymax`). By default,
{func}`~geoutils.Raster.crop` returns a new Raster.
The {func}`~geoutils.Raster.icrop` function accepts only a bounding box in pixel coordinates (colmin, rowmin, colmax, rowmax) and crop the raster accordingly.
By default, {func}`~geoutils.Raster.crop` and {func}`~geoutils.Raster.icrop` return a new Raster unless the inplace parameter is set to True, in which case the cropping operation is performed directly on the original raster object.
For more details, see the {ref}`specific section and function descriptions in the API<api-geo-handle>`.

### Example for {func}`~geoutils.Raster.crop`
```{code-cell} ipython3
# Crop raster to smaller bounds
rast_crop = rast.crop(bbox=(0.3, 0.3, 1, 1))
print(rast_crop.bounds)
```

### Example for {func}`~geoutils.Raster.icrop`
```{code-cell} ipython3
# Crop raster using pixel coordinates
rast_icrop = rast.icrop(bbox=(2, 2, 6, 6))
print(rast_icrop.bounds)
```

## Polygonize

Polygonizing a {class}`~geoutils.Raster` is done through the {func}`~geoutils.Raster.polygonize` function, which converts target pixels into a multi-polygon
{class}`~geoutils.Vector`.

```{note}
For a boolean {class}`~geoutils.Raster`, {func}`~geoutils.Raster.polygonize` implicitly targets `True` values and thus does not require target pixels.
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
For a boolean {class}`~geoutils.Raster`, {func}`~geoutils.Raster.proximity` implicitly targets `True` values and thus does not require target pixels.
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

## Filter
Filtering a {class}`~geoutils.Raster` is done through the {func}`~geoutils.Raster.filter` function.
The following filters are available:

| Filter Name | Description                                                                             | Typical Effect                                                  |
|:------------|:----------------------------------------------------------------------------------------|:----------------------------------------------------------------|
| `gaussian`  | Applies a Gaussian (blur) filter with a specified sigma.                                | Smooths the image, reduces noise while slightly blurring edges. |
| `median`    | Applies a median filter over a sliding window.                                          | Reduces noise while preserving edges better than Gaussian.      |
| `mean`      | Applies a mean (average) filter with a specified kernel size.                           | Smooths the image uniformly, reduces high-frequency noise.      |
| `max`       | Applies a maximum filter over a sliding window.                                         | Enhances bright regions, expands high-intensity areas.          |
| `min`       | Applies a minimum filter over a sliding window. | Suppresses bright regions, expands dark regions. |
| `distance`  | Removes pixels that deviate strongly from local neighborhood average (within a radius). | Removes outliers and anomalous values based on local context.   |

You can also pass a hand-made filter function for numpy arrays

```{code-cell} ipython3
# Filter the raster with a gaussian kernel
rast_filtered = rast.filter("gaussian", sigma=5)

# Filter the raster with a hand-made filter
def double_filter(arr: np.ndarray) -> np.ndarray:
    return arr * 2

rast_double = rast.filter(double_filter)
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

## Statistics

Statistics of a raster, optionally subsetting to an inlier mask, can be computed using {func}`~geoutils.Raster.get_stats`.

```{code-cell} ipython3
# Get mean, max and STD of the raster
rast.get_stats(["mean", "max", "std"])
```

A raster can also be quickly subsampled using {func}`~geoutils.Raster.subsample`, which can consider only valid values, and return either a point cloud or an
array:

```{code-cell} ipython3
# Get 500 random points
pc_sub = rast.subsample(500)
```

See {ref}`stats` for more details.

(mask-type)=
# The georeferenced raster mask (boolean {class}`~geoutils.Raster`)

A raster mask is a boolean {class}`~geoutils.Raster` (True or False).

While boolean data types are typically not supported in raster filetypes or in-memory operations, they are incredibly useful for various logical and
arithmetical operation in geospatial analysis, so GeoUtils facilitates their manipulation to support these operations natively and implicitly.

```{important}
Most raster file formats such a [GeoTIFFs](https://gdal.org/drivers/raster/gtiff.html) **do not support {class}`bool` array {class}`dtype<numpy.dtype>`
on-disk**, and **most of Rasterio functionalities also do not support {class}`bool` {class}`dtype<numpy.dtype>`**.

To address this, during opening, saving and other geospatial handling operations, raster masks are automatically converted to and from {class}`numpy.uint8`.
The {class}`~geoutils.Raster.nodata` of a boolean {class}`~geoutils.Raster` can now be defined to save to a file, and defaults to `255`.
```

## Open, cast and save

A raster mask can be opened from a file through instantiation with {class}`~geoutils.Raster` with the argument `is_mask=True`.

On opening, all data will be forced to a {class}`bool` {class}`numpy.dtype`.

```{code-cell} ipython3
import geoutils as gu

# Instantiate a mask from a filename on disk
filename_mask = gu.examples.get_path("exploradores_aster_dem")
mask = gu.Raster(filename_mask, load_data=True, is_mask=True)
mask
```

Raster masks are automatically cast by a logical comparison operation performed on a {class}`~geoutils.Raster` with either another
{class}`~geoutils.Raster`, a {class}`~numpy.ndarray` or a number.

```{code-cell} ipython3
# Instantiate a raster from disk
filename_rast = gu.examples.get_path("exploradores_aster_dem")
rast = gu.Raster(filename_rast, load_data=True)

# Which pixels are below 1500 m?
rast < 1500
```

See {ref}`core-py-ops` for more details.

## Create from {class}`~numpy.ndarray`

The class method {func}`geoutils.Raster.from_array` respects if the {class}`~numpy.ndarray` is of {class}`bool` {class}`~numpy.dtype`.

```{code-cell} ipython3
import rasterio as rio
import pyproj
import numpy as np

# Create a random 3 x 3 masked array
np.random.seed(42)
arr = np.random.randint(0, 2, size=(3, 3), dtype="bool")
mask = np.random.randint(0, 2, size=(3, 3), dtype="bool")
ma = np.ma.masked_array(data=arr, mask=mask)

# Cast to a mask from a boolean array through the Raster class
mask = gu.Raster.from_array(
        data = ma,
        transform = rio.transform.from_bounds(0, 0, 1, 1, 3, 3),
        crs = pyproj.CRS.from_epsg(4326),
        nodata = 255
    )
mask
```

## Create from {class}`~geoutils.Vector`

Raster masks can also be created from a {class}`~geoutils.Vector` using {class}`~geoutils.Vector.create_mask`, which rasterizes
all input geometries to a boolean array through {class}`~geoutils.Vector.rasterize`.

Georeferencing attributes to create the {class}`~geoutils.Raster` mask can also be passed individually, using `bounds`, `crs`, `xres` and `yres`.

```{code-cell} ipython3
# Open a vector of glacier outlines
filename_vect = gu.examples.get_path("exploradores_rgi_outlines")
vect = gu.Vector(filename_vect)

# Create mask using the raster as reference to match for bounds, resolution and projection
mask_outlines = vect.create_mask(rast)
mask_outlines
```

See {ref}`core-match-ref` for more details on the match-reference functionality.

## Arithmetic

Raster masks support Python's logical bitwise operators ({func}`~ <operator.invert>`, {func}`& <operator.and_>`, {func}`|<operator.or_>`,
{func}`^ <operator.xor>`) with other raster masks, and always output a raster mask.

```{code-cell} ipython3
# Combine masks
~mask | mask
```

## Indexing and assignment

Raster masks can be used for indexing and index assignment operations ({func}`[] <operator.getitem>`, {func}`[]= <operator.setitem>`) with a
{class}`Raster<geoutils.Raster>`.

```{important}
When indexing, a flattened {class}`~numpy.ma.MaskedArray` is returned with the indexed values of the {class}`~geoutils.Raster` **excluding those masked in its
{class}`~geoutils.Raster.data`'s {class}`~numpy.ma.MaskedArray`**.
```

```{code-cell} ipython3
# Index raster values on mask
rast[mask_outlines]
```

See {ref}`py-ops-indexing` for more details.

## Implicit polygonize, proximity and more

Raster masks have simplified class methods, when one or several attributes of the methods become implicit in the case of {class}`bool` data.

The {func}`~geoutils.Raster.polygonize` function is one of those, implicitly applying to the `True` values of the mask as target pixels. It outputs a
{class}`~geoutils.Vector` of the input mask.

```{code-cell} ipython3
# Polygonize mask
mask.polygonize()
```

The {func}`~geoutils.Raster.proximity` function is another method of {class}`~geoutils.Raster` implicitly applying to the `True` values of the mask as
target pixels. It outputs a {class}`~geoutils.Raster` of the distances to the input mask.

```{code-cell} ipython3
# Proximity to mask
mask.proximity()
```
