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

(core-py-ops)=
# Support of pythonic operators

GeoUtils integrates pythonic operators for shorter, more intuitive code, and to perform arithmetic and logical operations consistently.

These operators work on {class}`Rasters<geoutils.Raster>` much as they would on {class}`ndarrays<numpy.ndarray>`, with some more details.

## Arithmetic of {class}`~geoutils.Raster` classes

Arithmetic operators ({func}`+<operator.add>`, {func}`-<operator.sub>`, {func}`/<operator.truediv>`, {func}`//<operator.floordiv>`, {func}`*<operator.mul>`,
{func}`**<operator.pow>`, {func}`%<operator.mod>`) can be used on a {class}`~geoutils.Raster` in combination with any other {class}`~geoutils.Raster`,
{class}`~numpy.ndarray` or number.

For an operation with another {class}`~geoutils.Raster`, the georeferencing ({attr}`~geoutils.Raster.crs` and {attr}`~geoutils.Raster.transform`) must match.
For another {class}`~numpy.ndarray`, the {attr}`~geoutils.Raster.shape` must match. The operation always returns a {class}`~geoutils.Raster`.

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
{class}`~geoutils.Raster.nodata` values. Additionally, the {attr}`~geoutils.Raster.dtype` are also reconciled as they would for {class}`~numpy.ndarray`,
following [standard NumPy coercion rules](https://numpy.org/doc/stable/reference/generated/numpy.find_common_type.html).

## Logical comparisons cast to a raster mask

Logical comparison operators ({func}`==<operator.eq>`, {func}` != <operator.ne>`, {func}`>=<operator.ge>`, {func}`><operator.gt>`, {func}`<=<operator.le>`,
{func}`<<operator.lt>`) can be used on a {class}`~geoutils.Raster`, also in combination with any other {class}`~geoutils.Raster`, {class}`~numpy.ndarray` or
number.

Those operation always return a raster mask i.e. a {class}`~geoutils.Raster` with a boolean {class}`~numpy.ma.MaskedArray` as {class}`~geoutils.Raster.data`.

```{code-cell} ipython3
# Logical comparison with a number
mask = rast > 100
mask
```

```{note}
A boolean {class}`~geoutils.Raster`'s {attr}`~geoutils.Raster.data` remains a {class}`~numpy.ma.MaskedArray`. Therefore, it still maps invalid values
through its {attr}`~numpy.ma.MaskedArray.mask`, but has no associated {attr}`~geoutils.Raster.nodata`.
```

## Logical bitwise operations on raster masks

Logical bitwise operators ({func}`~ <operator.invert>`, {func}`& <operator.and_>`, {func}`| <operator.or_>`, {func}`^ <operator.xor>`) can be used to
combine a boolean {class}`~geoutils.Raster` with another boolean {class}`~geoutils.Raster`, and always output a boolean {class}`~geoutils.Raster`.

```{code-cell} ipython3
# Logical bitwise operation between masks
mask = (rast > 100) & ((rast % 2) == 0)
mask
```

(py-ops-indexing)=

## Indexing a {class}`~geoutils.Raster` with a raster mask

Finally, indexing and index assignment operations ({func}`[] <operator.getitem>`, {func}`[]= <operator.setitem>`) are both supported by
{class}`Rasters<geoutils.Raster>`.

For indexing, they can be passed either a boolean {class}`~geoutils.Raster` with the same georeferencing, or a boolean {class}`~numpy.ndarray` of the same
shape.
For assignment, either a {class}`~geoutils.Raster` with the same georeferencing, or any {class}`~numpy.ndarray` of the same shape is expected.

When indexing, a flattened {class}`~numpy.ma.MaskedArray` is returned with the indexed values of the boolean {class}`~geoutils.Raster` **excluding those masked
in its {class}`~geoutils.Raster.data`'s {class}`~numpy.ma.MaskedArray` (for instance, nodata values present during a previous logical comparison)**. To bypass this
behaviour, simply index without the mask using {attr}`Raster.data.data`.

```{code-cell} ipython3
# Indexing the raster with the previous mask
rast[mask]
```
