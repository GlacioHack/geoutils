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

(core-object-behaviour)=
# Object behaviour and inheritance

Much like Xarray objects, {class}`Rasters<geoutils.Raster>`, {class}`Vectors<geoutils.Vector>`, and
{class}`PointClouds<geoutils.PointCloud>` combine values with geospatial metadata, can defer reading file-backed data,
and preserve their geospatial type through many operations. Rasters and point clouds also expose their numeric values
through Python and NumPy interfaces, while vectors follow the behaviour of their GeoPandas dataframes.

(core-lazy-load)=
## Deferred I/O and implicit loading

**Deferred input/output** refers to operations that modify only **internal I/O metadata**, avoiding reading the data entirely and postponing loading.

Typical examples include {meth}`~RasterBase.crop`, {meth}`~RasterBase.copy`, and {meth}`~RasterBase.translate`,
which behave similarly as Xarray's {meth}`~xarray.DataArray.sel`, {meth}`~xarray.DataArray.copy`, or {meth}`~xarray.DataArray.assign_coords`.
File-backed vector and point cloud objects also defer spatial reads performed by {meth}`~VectorBase.crop`.

When using the Xarray {class}`rst <geoutils.RasterAccessor>` accessor, this behavior follows the **native Xarray deferred I/O model**. The {class}`~geoutils.Raster` class implements the
same behavior so that both APIs have consistent semantics.

An important aspect of **deferred I/O** is that it works with both **in-memory** (NumPy) and **scalable backends** (Dask), allowing
to extract parts of large files without any chunked or lazy considerations.

```{code-cell}
import geoutils as gu

# We open the dataset without Dask backend (same behaviour with Raster class)
filename_rast = gu.examples.get_path("exploradores_aster_dem")
ds = gu.open_raster(filename_rast)

# We crop the data
ds_cropped = ds.rst.icrop((0, 0, 100, 100))

# Neither input nor output raster are loaded yet
print(f"Is input raster loaded after cropping (deferred I/O)? {ds.rst.is_loaded}")
print(f"Is output raster loaded after cropping (deferred I/O)? {ds_cropped.rst.is_loaded}")
```

This behaviour pairs intrinsically with **implicit loading:** When an object is opened, only metadata is loaded.
Accessing {attr}`~RasterBase.data`, or calling operations that require the underlying array or geometries will **implicitly load the data into memory**.

```{code-cell}
# Is the above raster loaded?
print(f"Is raster loaded before data operation? {ds.rst.is_loaded}")

# We compute statistics, which loads the array
ds.rst.stats()

# The raster is now loaded
print(f"Is raster loaded after data operation? {ds.rst.is_loaded}")
```

Chunked and lazy execution are described in {ref}`scalability-concept`.

(core-py-ops)=
## Support of pythonic operators

GeoUtils integrates pythonic operators for shorter, more intuitive code, and to perform arithmetic and logical operations consistently.

These operators work on {class}`Rasters<geoutils.Raster>` much as they would on {class}`ndarrays<numpy.ndarray>`, with some more details.
For {class}`PointClouds<geoutils.PointCloud>`, they apply to the main {attr}`~PointCloudBase.data_column` in the same way.

### Arithmetic of {class}`~geoutils.Raster` classes

Arithmetic operators ({func}`+<operator.add>`, {func}`-<operator.sub>`, {func}`/<operator.truediv>`, {func}`//<operator.floordiv>`, {func}`*<operator.mul>`,
{func}`**<operator.pow>`, {func}`%<operator.mod>`) can be used on a {class}`~geoutils.Raster` in combination with any other {class}`~geoutils.Raster`,
{class}`~numpy.ndarray` or number.

For an operation with another {class}`~geoutils.Raster`, the georeferencing ({attr}`~RasterBase.crs` and {attr}`~RasterBase.transform`) must match.
For another {class}`~numpy.ndarray`, the {attr}`~RasterBase.shape` must match. The operation always returns a {class}`~geoutils.Raster`.

```{code-cell} ipython3
import geoutils as gu
import rasterio as rio
import pyproj
import numpy as np

# Create a random 3 x 3 masked array
np.random.seed(42)
arr = np.random.randint(0, 255, size=(3, 3), dtype="uint8")
mask = np.random.randint(0, 2, size=(3, 3), dtype="bool")
ma = np.ma.masked_array(data=arr, mask=mask)

# Create an example raster
rast = gu.Raster.from_array(
       data = ma,
       transform = rio.transform.from_bounds(0, 0, 1, 1, 3, 3),
       crs = pyproj.CRS.from_epsg(4326),
       nodata = 255
    )

rast
```

```{code-cell} ipython3
# Arithmetic with a number
rast + 1
```

```{code-cell} ipython3
# Arithmetic with an array
rast / arr

```
```{code-cell} ipython3
# Arithmetic with a raster
rast - (rast**0.5)
```

If an unmasked {class}`~numpy.ndarray` is passed, it will internally be cast into a {class}`~numpy.ma.MaskedArray` to respect the propagation of
{attr}`~RasterBase.nodata` values. Additionally, the {attr}`~RasterBase.dtype` are also reconciled as they would for {class}`~numpy.ndarray`,
following [standard NumPy promotion rules](https://numpy.org/doc/stable/reference/arrays.promotion.html).

### Logical comparisons cast to a raster mask

Logical comparison operators ({func}`==<operator.eq>`, {func}` != <operator.ne>`, {func}`>=<operator.ge>`, {func}`><operator.gt>`, {func}`<=<operator.le>`,
{func}`<<operator.lt>`) can be used on a {class}`~geoutils.Raster`, also in combination with any other {class}`~geoutils.Raster`, {class}`~numpy.ndarray` or
number.

Those operations always return a raster mask, i.e. a {class}`~geoutils.Raster` with a boolean {class}`~numpy.ma.MaskedArray` as {attr}`~RasterBase.data`.

```{code-cell} ipython3
# Logical comparison with a number
mask = rast > 100
mask
```

```{note}
A boolean {class}`~geoutils.Raster`'s {attr}`~RasterBase.data` remains a {class}`~numpy.ma.MaskedArray`. Therefore, it still maps invalid values
through its {attr}`~numpy.ma.MaskedArray.mask`, but has no associated {attr}`~RasterBase.nodata`.
```

### Logical bitwise operations on raster masks

Logical bitwise operators ({func}`~ <operator.invert>`, {func}`& <operator.and_>`, {func}`| <operator.or_>`, {func}`^ <operator.xor>`) can be used to
combine a boolean {class}`~geoutils.Raster` with another boolean {class}`~geoutils.Raster`, and always output a boolean {class}`~geoutils.Raster`.

```{code-cell} ipython3
# Logical bitwise operation between masks
mask = (rast > 100) & ((rast % 2) == 0)
mask
```

(py-ops-indexing)=

### Indexing a {class}`~geoutils.Raster` with a raster mask

Finally, indexing and index assignment operations ({func}`[] <operator.getitem>`, {func}`[]= <operator.setitem>`) are both supported by
{class}`Rasters<geoutils.Raster>`.

For indexing, they can be passed either a boolean {class}`~geoutils.Raster` with the same georeferencing, or a boolean {class}`~numpy.ndarray` of the same
shape.
For assignment, either a {class}`~geoutils.Raster` with the same georeferencing, or any {class}`~numpy.ndarray` of the same shape is expected.

When indexing, a flattened {class}`~numpy.ma.MaskedArray` is returned with the indexed values of the boolean {class}`~geoutils.Raster` **excluding those masked
in its {attr}`~RasterBase.data`'s {class}`~numpy.ma.MaskedArray` (for instance, nodata values present during a previous logical comparison)**. To bypass this
behaviour, simply index without the mask using {attr}`Raster.data.data`.

```{code-cell} ipython3
# Indexing the raster with the previous mask
rast[mask]
```

(core-array-funcs)=

## Masked-array NumPy interface

NumPy possesses an [array interface](https://numpy.org/doc/stable/reference/arrays.interface.html) that allows to properly map their functions on objects
that depend on {class}`ndarrays<numpy.ndarray>`.

GeoUtils utilizes this interface to work with all {class}`Rasters<geoutils.Raster>` and their subclasses. It also
applies the same functions to the main {attr}`~PointCloudBase.data_column` of {class}`PointClouds<geoutils.PointCloud>`.

### Universal functions

A first category of NumPy functions supported by {class}`Rasters<geoutils.Raster>` through the array interface is that of
[universal functions](https://numpy.org/doc/stable/reference/ufuncs.html), which operate on {class}`ndarrays<numpy.ndarray>` in an element-by-element
fashion. Examples of such functions are {func}`~numpy.add`, {func}`~numpy.absolute`, {func}`~numpy.isnan` or {func}`~numpy.sin`, and they number at more
than 90.

Universal functions can take one or two inputs, and return one or two outputs. Through GeoUtils, as long as one of the two inputs is a {class}`Rasters<geoutils.Raster>`,
the output will be a {class}`~geoutils.Raster`. If there is a second input, it can be a {class}`~geoutils.Raster` or {class}`~numpy.ndarray` with
matching georeferencing or shape, respectively.

These functions inherently support the casting of different {attr}`~RasterBase.dtype` and values masked by {attr}`~RasterBase.nodata` in the
{class}`~numpy.ma.MaskedArray`.

Below, we reuse the same example created in {ref}`core-py-ops`.

```{code-cell} ipython3
# Universal function with a single input and output
np.sin(rast)
```

```{code-cell} ipython3
# Universal function with a two inputs and single output
np.add(arr, rast)
```

```{code-cell} ipython3
# Universal function with a single input and two outputs
np.modf(rast)
```

Similar to with Python operators, NumPy's [logical comparison functions](https://numpy.org/doc/stable/reference/ufuncs.html#comparison-functions) cast
{class}`Rasters<geoutils.Raster>` to a boolean {class}`~geoutils.Raster`, a raster mask.

```{code-cell} ipython3
# Universal function with a single input and two outputs
np.greater(rast, rast + np.random.normal(size=np.shape(arr)))
```

### Array functions

The second and last category of NumPy array functions supported by {class}`Rasters<geoutils.Raster>` through the array interface is that of array functions,
which are all other non-universal functions that can be applied to an array. Those function always modify the dimensionality of the output, such as
{func}`~numpy.mean`, {func}`~numpy.count_nonzero` or {func}`~numpy.nanmax`. Consequently, the output is the same as it would be with {class}`ndarrays<numpy.ndarray>`.


```{code-cell} ipython3
# Traditional mathematical function
np.max(rast)
```

```{code-cell} ipython3
# Specify an axis for reduction
np.count_nonzero(rast, axis=1)
```

Not all array functions are supported, however. GeoUtils supports nearly all [mathematical functions](https://numpy.org/doc/stable/reference/routines.math.html),
[masked-array functions](https://numpy.org/doc/stable/reference/routines.ma.html) and [logical functions](https://numpy.org/doc/stable/reference/routines.logic.html).
A full list of supported array function is available in {attr}`geoutils.raster.handled_array_funcs`.

### Respecting masked values

There are two ways to compute statistics on {class}`Rasters<geoutils.Raster>` while respecting masked values:

1. Use any NumPy core function (`np.func`) directly on the {class}`~geoutils.Raster` (this includes NaN functions `np.nanfunc`),
2. Use any NumPy masked-array function (`np.ma.func`) on {attr}`Raster.data<RasterBase.data>`.

```{code-cell} ipython3
# Numpy core function applied to the raster
np.median(rast)
```

```{code-cell} ipython3
# Numpy NaN function applied to the raster
np.nanmedian(rast)
```

```{code-cell} ipython3
# Masked-array function on the data
np.ma.median(rast.data)
```

If a NumPy core function raises an error (e.g., {func}`numpy.percentile`), {attr}`~RasterBase.nodata` values might not be respected. In this case, use the NaN
function on the {class}`~geoutils.Raster`.


```{note}
Masked-array functions such as `np.ma.median` are not recognized when applied directly to a
{class}`~geoutils.Raster`. Apply them to {attr}`~RasterBase.data` instead.
```

(core-inheritance)=

## Inheritance to DEMs and beyond

Inheritance is practical to naturally pass down parent methods and attributes to child classes.

Many subtypes of {class}`Rasters<geoutils.Raster>` geospatial data exist that require additional attributes and methods, yet might benefit from methods
implemented in GeoUtils.

### Overview of {class}`~geoutils.Raster` inheritance

Current {class}`~geoutils.Raster` inheritance extends into other packages, such as [xDEM](https://xdem.readthedocs.io/)
for analyzing digital elevation models.

```{eval-rst}
.. inheritance-diagram:: geoutils.raster.raster
    :top-classes: geoutils.raster.raster.Raster
```

```{note}
The {class}`~xdem.DEM` class re-implements all methods of [gdalDEM](https://gdal.org/programs/gdaldem.html) (and more) to derive topographic attributes
(hillshade, slope, aspect, etc), coded directly in Python for scalability and tested to yield the exact same results.
Among others, it also adds a {attr}`~xdem.DEM.vcrs` property to consistently manage vertical referencing (ellipsoid, geoids).

If you are DEM-enthusiastic, **[check-out our sister package xDEM](https://xdem.readthedocs.io/) for digital elevation models.**
```

### And beyond

Many types of geospatial data can be viewed as a subclass of {class}`Rasters<geoutils.Raster>`, which have more attributes and require their own methods:
**spectral images**, **velocity fields**, **phase difference maps**, etc...

If you are interested to build your own subclass of {class}`~geoutils.Raster`, you can take example of the structure of {class}`xdem.DEM`.
Then, just add any of your own attributes and methods, and overload parent methods if necessary! Don't hesitate to reach out on our
GitHub if you have a subclassing project.
