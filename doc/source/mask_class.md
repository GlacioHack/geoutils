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
(mask-class)=

# The georeferenced mask ({class}`~geoutils.Mask`)

Below, a summary of the {class}`~geoutils.Mask` object and its methods.

## Object definition and attributes

A {class}`~geoutils.Mask` is a subclass of {class}`~geoutils.Raster` that contains **three main attributes**:

1. a {class}`numpy.ma.MaskedArray` as {attr}`~geoutils.Raster.data` of {class}`~numpy.boolean` {class}`~numpy.dtype`,
2. an [{class}`affine.Affine`](https://rasterio.readthedocs.io/en/stable/topics/migrating-to-v1.html#affine-affine-vs-gdal-style-geotransforms) as {attr}`~geoutils.Raster.transform`, and
3. a {class}`pyproj.crs.CRS` as {attr}`~geoutils.Raster.crs`.

A {class}`~geoutils.Mask` also inherits the same derivative attributes as a {class}`~geoutils.Raster`.

```{note}
There is no {class}`~geoutils.Raster.nodata` value defined in a {class}`~geoutils.Mask`, as it only take binary values. However, the
{class}`numpy.ma.MaskedArray` still has a {class}`~geoutils.Raster.data.mask` for invalid values.
```

## Open and save

{class}`Masks<geoutils.Mask>` can be created from files through instantiation with {class}`~geoutils.Mask`, and inherit the {func}`~geoutils.Raster.save`
method from {class}`~geoutils.Raster`.

```{important}
Most raster file formats such a [GeoTIFFs](https://gdal.org/drivers/raster/gtiff.html) **do not support {class}`bool` array {class}`dtype<numpy.dtype>`
on-disk**, and **most of Rasterio functionalities also do not support {class}`bool` {class}`dtype<numpy.dtype>`**.

To address this, during opening, saving and geospatial handling operations, {class}`Masks<geoutils.Mask>` are automatically converted to and from {class}`numpy.uint8`.
The {class}`~geoutils.Raster.nodata` of a {class}`~geoutils.Mask` can now be defined to save to a file, and defaults to `255`.
```

On opening, all data will be forced to a {class}`bool` {class}`numpy.dtype`.

```{code-cell} ipython3
import geoutils as gu

# Instantiate a mask from a filename on disk
filename_mask = gu.examples.get_path("exploradores_aster_dem")
mask = gu.Mask(filename_mask, load_data=True)
mask
```

## Cast from {class}`~geoutils.Raster`

{class}`Masks<geoutils.Mask>` are automatically cast by a logical comparison operation performed on a {class}`~geoutils.Raster` with either another
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

The class method {func}`geoutils.Raster.from_array`, inherited by the {class}`~geoutils.Mask` subclass, automatically casts to a {class}`~geoutils.Mask` if
the input {class}`~numpy.ndarray` is of {class}`bool` {class}`~numpy.dtype`.

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

When creating with {func}`geoutils.Mask.from_array`, any input {class}`~numpy.ndarray` will be forced to a {class}`bool` {class}`numpy.dtype`.

```{code-cell} ipython3
# Create a mask from array directly from the Mask class
mask = gu.Mask.from_array(
        data = ma.astype("float32"),
        transform = rio.transform.from_bounds(0, 0, 1, 1, 3, 3),
        crs = pyproj.CRS.from_epsg(4326),
        nodata = 255
    )
mask
```

## Create from {class}`~geoutils.Vector`

{class}`Masks<geoutils.Mask>` can also be created from a {class}`~geoutils.Vector` using {class}`~geoutils.Vector.create_mask`, which rasterizes all input
geometries to a boolean array through {class}`~geoutils.Vector.rasterize`.

Georeferencing attributes to create the {class}`~geoutils.Mask` can also be passed individually, using `bounds`, `crs`, `xres` and `yres`.

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

{class}`Masks<geoutils.Mask>` support Python's logical bitwise operators ({func}`~ <operator.invert>`, {func}`& <operator.and_>`, {func}`|<operator.or_>`,
{func}`^ <operator.xor>`) with other {class}`Masks<geoutils.Mask>`, and always output a {class}`~geoutils.Mask`.

```{code-cell} ipython3
# Combine masks
~mask | mask
```

## Indexing and assignment

{class}`Masks<geoutils.Mask>` can be used for indexing and index assignment operations ({func}`[] <operator.getitem>`, {func}`[]= <operator.setitem>`) with
{class}`Rasters<geoutils.Raster>`.

```{important}
When indexing, a flattened {class}`~numpy.ma.MaskedArray` is returned with the indexed values of the {class}`~geoutils.Mask` **excluding those masked in its
{class}`~geoutils.Raster.data`'s {class}`~numpy.ma.MaskedArray`**.
```

```{code-cell} ipython3
# Index raster values on mask
rast[mask_outlines]
```

See {ref}`py-ops-indexing` for more details.

(mask-class-poly-overloaded)=

## Polygonize (overloaded from {class}`~geoutils.Raster`)

{class}`Masks<geoutils.Mask>` have simplified class methods overloaded from {class}`Rasters<geoutils.Raster>` when one or several attributes of the methods
become implicit in the case of {class}`bool` data.

The {func}`~geoutils.Mask.polygonize` function is one of those, implicitly applying to the `True` values of the mask as target pixels. It outputs a
{class}`~geoutils.Vector` of the input mask.

```{code-cell} ipython3
# Polygonize mask
mask.polygonize()
```

(mask-class-prox-overloaded)=

## Proximity (overloaded from {class}`~geoutils.Raster`)

The {func}`~geoutils.Mask.proximity` function is another overloaded method of {class}`~geoutils.Raster` implicitly applying to the `True` values of the mask as
target pixels. It outputs a {class}`~geoutils.Raster` of the distances to the input mask.

```{code-cell} ipython3
# Proximity to mask
mask.proximity()
```
