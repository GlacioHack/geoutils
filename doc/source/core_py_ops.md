---
file_format: mystnb
kernelspec:
  name: geoutils
---

(core-py-ops)=
# Support of pythonic operators

GeoUtils integrates most of Python's operators for shorter, more intuitive code to consistently perform arithmetic and logical operations.
 
Pythonic operators work on {class}`~geoutils.Raster` much as they would on {class}`~numpy.ndarray`, with some more details.

## Arithmetic of {class}`~geoutils.Raster` classes

Arithmetic operators (+, -, /, //, *, **, %) can be used on a {class}`~geoutils.Raster` in combination with any other {class}`~geoutils.Raster`, {class}`~numpy.
ndarray` or number.

For an operation with another {class}`~geoutils.Raster`, the georeferencing ({attr}`~geoutils.Raster.crs` and {attr}`~geoutils.Raster.transform`) must match. 
For another {class}`~numpy.ndarray`, the {attr}`~geoutils.Raster.shape` must match. The operation always returns a {class}`~geoutils.Raster`.

```{code-cell} ipython3
import geoutils as gu
import rasterio as rio
import numpy as np

# Create a random 3 x 3 masked array
np.random.seed(42)
arr = np.random.randint(0, 255, size=(3, 3), dtype="uint8")
mask = np.random.randint(0, 2, size=(3, 3), dtype="bool")
ma = np.ma.masked_array(data=arr, mask=mask)

# Create an example Raster with only a transform
raster = gu.Raster.from_array(
        data = ma,
        transform = rio.transform.from_bounds(0, 0, 1, 1, 3, 3),
        crs = None
    )

raster
```

```{code-cell} ipython3
# Arithmetic with a number
raster + 1
```

```{code-cell} ipython3
# Arithmetic with an array
raster / arr

```
```{code-cell} ipython3
# Arithmetic with a raster
raster - (raster**0.5)
```

If an unmasked {class}`~numpy.ndarray` is passed, it will internally be cast into a {class}`~numpy.ma.MaskedArray` to respect the propagation of 
{class}`~geoutils.Raster.nodata` values. Additionally, the {attr}`~geoutils.Raster.dtypes` are also reconciled as they would for {class}`~numpy.ndarray`, 
following [standard NumPy coercion rules](https://numpy.org/doc/stable/reference/generated/numpy.find_common_type.html).

## Logical comparisons cast to {class}`~geoutils.Mask`

Logical comparison operators (==, !=, >=, >, <=, <) can be used on a {class}`~geoutils.Raster`, also in combination with any other {class}`~geoutils.Raster`, 
{class}`~numpy.ndarray` or number.

Those operation always return a {class}`~geoutils.Mask`, a subclass of {class}`~geoutils.Raster` with a boolean {class}`~numpy.ma.MaskedArray` 
as {class}`~geoutils.Raster.data`.

```{code-cell} ipython3
# Logical comparison with a number
mask = raster > 100
mask
```

```{note}
A {class}`~geoutils.Mask`'s {attr}`~geoutils.Raster.data` remains a {class}`~numpy.ma.MaskedArray`. Therefore, it still maps unvalid values through its 
{attr}`~numpy.ma.MaskedArray.mask`, but has no associated {attr}`~geoutils.Raster.nodata`.
```

## Logical bitwise operations on {class}`~geoutils.Mask`

Logical bitwise operators (~, &, |, ^) can be used to combine a {class}`~geoutils.Mask` with another {class}`~geoutils.Mask`, and always output a {class}`~geoutils.Mask`.

```{code-cell} ipython3
# Logical bitwise operation between masks
mask = (raster > 100) & ((raster % 2) == 0)
mask
```

## Indexing a {class}`~geoutils.Raster` with a {class}`~geoutils.Mask`