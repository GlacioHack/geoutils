---
file_format: mystnb
kernelspec:
  name: geoutils
---
(quick-start)=

# Quick start

The following presents a descriptive example show-casing all core aspects of GeoUtils.

For more details, refer to the {ref}`core-index`, {ref}`rasters-index` or {ref}`vectors-index` pages.

To find an example about a specific functionality, jump to {ref}`quick-gallery`.

## The core {class}`~geoutils.Raster` and {class}`~geoutils.Vector` classes

In GeoUtils, geospatial handling is object-based and revolves around {class}`~geoutils.Raster` and {class}`~geoutils.Vector`.
These link to either **on-disk** or **in-memory** datasets, opened by instantiating the class.

```{code-cell} ipython3
import geoutils as gu

# Examples files: infrared band of Landsat and glacier outlines
filename_rast = gu.examples.get_path("everest_landsat_b4")
filename_vect = gu.examples.get_path("everest_rgi_outlines")

# Open files
rast = gu.Raster(filename_rast)
vect = gu.Vector(filename_vect)
```

A {class}`~geoutils.Raster` is a composition class with four main attributes: a {class}`numpy.ma.MaskedArray` as {attr}`~geoutils.Raster.data`, a
{class}`pyproj.crs.CRS` as {attr}`~geoutils.Raster.crs`, an {class}`affine.Affine` as {attr}`~geoutils.Raster.transform`, and a {class}`float` or 
{class}`int` as {attr}`~geoutils.Raster.nodata`.


```{code-cell} ipython3
# The opened raster
rast
```

```{important}
When a file exists on disk, {class}`~geoutils.Raster` is linked to a {class}`rasterio.io.DatasetReader` object for loading the metadata. The array will be
**loaded in-memory implicitly** when {attr}`~geoutils.Raster.data` is required by an operation.

See {ref}`core-lazy-load` for more details.
```

A {class}`~geoutils.Vector` is a composition class with a single main attribute: a {class}`~geopandas.GeoDataFrame` as {attr}`~geoutils.Vector.ds`, for which
most methods are wrapped directly into {class}`~geoutils.Vector`.

```{code-cell} ipython3
:tags: [hide-output]
# The opened vector
vect
```

All other attributes are derivatives of those main attributes, or of the filename on disk. Attributes of {class}`~geoutils.Raster` and
{class}`~geoutils.Vector` update with geospatial operations on themselves.


## Handling and match-reference

In GeoUtils, geospatial handling operations are based on class methods, such as {func}`~geoutils.Raster.crop` or {func}`~geoutils.Raster.reproject`.

For convenience and consistency, nearly all of these methods can be passed solely another {class}`~geoutils.Raster` or {class}`~geoutils.Vector` as a
**reference to match** during the operation. A **reference {class}`~geoutils.Vector`** enforces a matching of {attr}`~geoutils.Vector.bounds` and/or
{attr}`~geoutils.Vector.crs`, while a **reference {class}`~geoutils.Raster`** can also enforce a matching of {attr}`~geoutils.Raster.res`, depending on the nature of the operation.


```{code-cell} ipython3
:tags: [hide-output]
# Crop raster to vector's extent
rast.crop(vect)
# Print info of cropped raster
print(rast.info())
```

```{margin}
<sup>1</sup>The names of geospatial handling methods is largely based on [GDAL and OGR](https://gdal.org/)'s, with the notable exception of {func}`~geoutils.Vector.reproject` that better applies to vectors than `warp`.
```

Additionally, in GeoUtils, **methods that apply to the same georeferencing attributes have consistent naming**<sup>1</sup> across {class}`~geoutils.Raster` and {class}`~geoutils.Vector`.

A {func}`~geoutils.Raster.reproject` involves a change in {attr}`~geoutils.Raster.crs` or {attr}`~geoutils.Raster.transform`, while a {func}`~geoutils.Raster.crop` only involves a change
in {attr}`~geoutils.Raster.bounds`. Using {func}`~geoutils.Raster.polygonize` allows to generate a {class}`~geoutils.Vector` from a {class}`~geoutils.Raster`,
and the other way around for {func}`~geoutils.Vector.rasterize`.

```{list-table}
   :widths: 30 30 30
   :header-rows: 1

   * - **Class**
     - {class}`~geoutils.Raster`
     - {class}`~geoutils.Vector`
   * - Warping/Reprojection
     - {func}`~geoutils.Raster.reproject`
     - {func}`~geoutils.Vector.reproject`
   * - Cropping/Clipping
     - {func}`~geoutils.Raster.crop`
     - {func}`~geoutils.Vector.crop`
   * - Rasterize/Polygonize
     - {func}`~geoutils.Raster.polygonize`
     - {func}`~geoutils.Vector.rasterize`
```

All methods can be also be passed any number of georeferencing arguments such as {attr}`~geoutils.Raster.shape` or {attr}`~geoutils.Raster.res`, and will
naturally deduce others from the input {class}`~geoutils.Raster` or {class}`~geoutils.Vector`, much as in [GDAL](https://gdal.org/)'s command line.


## Higher-level analysis tools

GeoUtils also implements higher-level geospatial analysis tools for both {class}`Rasters<geoutils.Raster>` and {class}`Vectors<geoutils.Vector>`. For
example, one can compute the distance to a {class}`~geoutils.Vector` geometry, or to target pixels of a {class}`~geoutils.Raster`, using
{func}`~geoutils.Vector.proximity`.

As with the geospatial handling functions previously listed, many analysis functions can take a {class}`~geoutils.Raster` or {class}`~geoutils.Vector` as a
**reference to utilize** during the operation. In the case of {func}`~geoutils.Vector.proximity`, passing a {class}`~geoutils.Raster` serves as a reference
for the georeferenced grid on which to compute the distances.

```{code-cell} ipython3
# Compute proximity to vector on raster's grid
rast_proximity_to_vec = vect.proximity(rast)
```

```{note}
Right now, the array {attr}`~geoutils.Raster.data` of `rast` is still not loaded. Applying {func}`~geoutils.Raster.crop` does not yet require loading,
and `rast`'s metadata is sufficient to provide a georeferenced grid for {func}`~geoutils.Vector.proximity`. The array will only be loaded when necessary.
```

## Quick plotting

To facilitate the analysis process, GeoUtils includes quick plotting tools that support multiple colorbars and implicitly add layers to the current axis.
Those are wrapped from {func}`rasterio.plot.show` and {func}`geopandas.GeoDataFrame.plot`, and relay any argument passed.

```{seealso}
GeoUtils' plotting tools only aim to smooth out the most common hassles when quickly plotting raster and vectors.

For advanced plotting tools to create "publication-quality" figures, see [Cartopy](https://scitools.org.uk/cartopy/docs/latest/) or
[GeoPlot](https://residentmario.github.io/geoplot/index.html).
```

The plotting functionality is named {func}`~geoutils.Raster.show` everywhere, for consistency. Here again, a {class}`~geoutils.Raster` or
{class}`~geoutils.Vector` can be passed as a **reference to match** to ensure all data is displayed on the same grid and projection.

```{code-cell} ipython3
# Plot proximity to vector
rast_proximity_to_vec = vect.proximity(rast)
rast_proximity_to_vec.show(cbar_title="Distance to glacier outline")
vect.show(rast_proximity_to_vec, fc="none")
```

## Pythonic arithmetic and NumPy interface

All {class}`~geoutils.Raster` objects support Python arithmetic ({func}`+<operator.add>`, {func}`-<operator.sub>`, {func}`/<operator.truediv>`, {func}`//<operator.floordiv>`, {func}`*<operator.mul>`,
{func}`**<operator.pow>`, {func}`%<operator.mod>`) with any other {class}`~geoutils.Raster`, {class}`~numpy.ndarray` or
number. With another {class}`~geoutils.Raster`, the georeferencing must match, while only the shape with a {class}`~numpy.ndarray`.

```{code-cell} ipython3
# Add 1 to the raster array
rast += 1
```

Additionally, the class {class}`~geoutils.Raster` possesses a NumPy masked-array interface that allows to apply to it any [NumPy universal function](https://numpy.org/doc/stable/reference/ufuncs.html) and
most other NumPy array functions, while logically casting {class}`dtypes<numpy.dtype>` and respecting {attr}`~geoutils.Raster.nodata` values.

```{code-cell} ipython3
# Apply a normalization to the raster
import numpy as np
rast = (rast - np.min(rast)) / (np.max(rast) - np.min(rast))
```

## Casting to {class}`~geoutils.Mask`, indexing and overload

All {class}`~geoutils.Raster` classes also support Python logical comparison operators ({func}`==<operator.eq>`, {func}` != <operator.ne>`, {func}`>=<operator.ge>`, {func}`><operator.gt>`, {func}`<=<operator.le>`,
{func}`<<operator.lt>`), or more complex NumPy logical functions. Those operations automatically casts them into a {class}`~geoutils.Mask`, a subclass of {class}`~geoutils.Raster`.

```{code-cell} ipython3
# Get mask of an AOI: infrared index above 0.7, at least 200 m from glaciers
mask_aoi = np.logical_and(rast > 0.7, rast_proximity_to_vec > 200)
```

Masks can then be used for indexing a {class}`~geoutils.Raster`, which returns a {class}`~numpy.ma.MaskedArray` of indexed values.

```{code-cell} ipython3
# Index raster with mask to extract a 1-D array
values_aoi = rast[mask_aoi]
```

Masks also have simplified, overloaded {class}`~geoutils.Raster` methods due to their boolean {class}`dtypes<numpy.dtype>`. Using {func}`~geoutils.Raster.polygonize` with a
{class}`~geoutils.Mask` is straightforward, for instance, to retrieve a {class}`~geoutils.Vector` of the area-of-interest:

```{code-cell} ipython3
# Polygonize areas where mask is True
vect_aoi = mask_aoi.polygonize()
```

```{code-cell} ipython3
# Plot result
rast.show(cmap='Reds', cbar_title='Normalized infrared')
vect_aoi.show(fc='none', ec='k', lw=0.5)
```

## Saving to file

Finally, for saving a {class}`~geoutils.Raster` or {class}`~geoutils.Vector` to file, simply call the {func}`~geoutils.Raster.save` function.

```{code-cell} ipython3
# Save our AOI vector
vect_aoi.save("myaoi.gpkg")
```

```{code-cell} ipython3
:tags: [remove-cell]
import os
os.remove("myaoi.gpkg")
```

## Parsing metadata with {class}`~geoutils.SatelliteImage`

In our case, `rast` would be better opened using the {class}`~geoutils.Raster` subclass {class}`~geoutils.SatelliteImage` instead, which tentatively parses
metadata recognized from the filename or auxiliary files.

```{code-cell} ipython3
# Name of the image we used
import os
print(os.path.basename(filename_rast))
```

```{code-cell} ipython3
# Open while parsing metadata
rast = gu.SatelliteImage(filename_rast, silent=False)
```

```{admonition} Wrap-up
In a few lines, we:
 - **easily handled georeferencing** operations on rasters and vectors,
 - performed numerical calculations **inherently respecting invalid data**,
 - **naturally casted to a mask** from a logical operation on raster, and
 - **straightforwardly vectorized a mask** by harnessing overloaded subclass methods.

**Our result:** a vector of high infrared absorption indexes at least 200 meters away from glaciers
near Everest, which likely corresponds to **perennial snowfields**.

Otherwise, for more **hands-on** examples, explore GeoUtils' gallery of examples!
```

(quick-gallery)=
## More examples

To dive into more illustrated code, explore our gallery of examples that is composed of:
- An {ref}`examples-io` section on opening, saving, loading, importing and exporting,
- An {ref}`examples-handling` section on geotransformations (crop, reproject) and raster-vector interfacing,
- An {ref}`examples-analysis` section on analysis tools and raster numerics.

See also the full concatenated list of examples below.

```{eval-rst}
.. minigallery:: geoutils.Raster
    :add-heading: Examples using rasters and vectors
```

