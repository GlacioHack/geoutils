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
(core-array-funcs)=

# Masked-array NumPy interface

NumPy possesses an [array interface](https://numpy.org/doc/stable/reference/arrays.interface.html) that allows to properly map their functions on objects
that depend on {class}`ndarrays<numpy.ndarray>`.

GeoUtils utilizes this interface to work with all {class}`Rasters<geoutils.Raster>` and their subclasses.

## Universal functions

A first category of NumPy functions supported by {class}`Rasters<geoutils.Raster>` through the array interface is that of
[universal functions](https://numpy.org/doc/stable/reference/ufuncs.html), which operate on {class}`ndarrays<numpy.ndarray>` in an element-by-element
fashion. Examples of such functions are {func}`~numpy.add`, {func}`~numpy.absolute`, {func}`~numpy.isnan` or {func}`~numpy.sin`, and they number at more
than 90.

Universal functions can take one or two inputs, and return one or two outputs. Through GeoUtils, as long as one of the two inputs is a {class}`Rasters<geoutils.Raster>`,
the output will be a {class}`~geoutils.Raster`. If there is a second input, it can be a {class}`~geoutils.Raster` or {class}`~numpy.ndarray` with
matching georeferencing or shape, respectively.

These functions inherently support the casting of different {attr}`~geoutils.Raster.dtype` and values masked by {attr}`~geoutils.Raster.nodata` in the
{class}`~numpy.ma.MaskedArray`.

Below, we reuse the same example created in {ref}`core-py-ops`.

```{code-cell} ipython3
:tags: [hide-input, hide-output]

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

## Array functions

The second and last category of NumPy array functions supported by {class}`Rasters<geoutils.Raster>` through the array interface is that of array functions,
which are all other non-universal functions that can be applied to an array. Those function always modify the dimensionality of the output, such as
{func}`~numpy.mean`, {func}`~numpy.count_nonzero` or {func}`~numpy.nanmax`. Consequently, the output is the same as it would be with {class}`ndarrays<numpy.ndarray>`.


```{code-cell} ipython3
# Traditional mathematical function
np.max(rast)
```

```{code-cell} ipython3
# Expliciting an axis for reduction
np.count_nonzero(rast, axis=1)
```

Not all array functions are supported, however. GeoUtils supports nearly all [mathematical functions](https://numpy.org/doc/stable/reference/routines.math.html),
[masked-array functions](https://numpy.org/doc/stable/reference/routines.ma.html) and [logical functions](https://numpy.org/doc/stable/reference/routines.logic.html).
A full list of supported array function is available in {attr}`geoutils.raster.handled_array_funcs`.

## Respecting masked values

There are two ways to compute statistics on {class}`Rasters<geoutils.Raster>` while respecting masked values:

1. Use any NumPy core function (`np.func`) directly on the {class}`~geoutils.Raster` (this includes NaN functions `np.nanfunc`),
2. Use any NumPy masked-array function (`np.ma.func`) on {attr}`Raster.data<geoutils.Raster.data>`.

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

If a NumPy core function raises an error (e.g., {func}`numpy.percentile`), {class}`~geoutils.Raster.nodata` values might not be respected. In this case, use the NaN
function on the {class}`~geoutils.Raster`.


```{note}
Unfortunately, masked-array functions `np.ma.func` cannot be recognized yet if applied directly to a {class}`~geoutils.Raster`, but **this should come
soon** as related interfacing is in the works in NumPy!
```
