---
file_format: mystnb
kernelspec:
  name: geoutils
---
(quick-start)=

# Quick start

The following presents how to quickly get started with GeoUtils, show-casing examples on different core aspects of the package.

For more details, refer to the {ref}`core-index`, {ref}`rasters-index` or {ref}`vectors-index` pages.

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

A {class}`~geoutils.Raster` is a composition class with four main attributes: a {class}`~numpy.ma.MaskedArray` as `.data`, a {class}`~pyproj.crs.CRS` as `.crs`, 
an {class}`~affine.Affine` as `.transform`, and a {class}`float` or {class}`int` as `.nodata`. When a file exists on disk, {class}`~geoutils.Raster` is 
linked to a {class}`rasterio.io.DatasetReader` object for loading the metadata, and the array at the appropriate time.

```{code-cell} ipython3
:tags: [hide-output]
# The opened raster
rast
```

A {class}`~geoutils.Vector` is a composition class with a single main attribute: a {class}`~geopandas.GeoDataFrame` as `.ds`, for which most methods are 
wrapped directly into {class}`~geoutils.Vector`. 

```{code-cell} ipython3
:tags: [hide-output]
# The opened vector
vect
```

All other attributes are derivatives of those main attributes, or of the filename on disk. Attributes of {class}`~geoutils.Raster` and 
{class}`~geoutils.Vector` update with geospatial operations on themselves. 

## Geospatial handling and match-reference

Geospatial operations are based on class methods, such as {func}`geoutils.Raster.crop` or {func}`geoutils.Vector.proximity`. Nearly all of these methods can be 
passed solely another {class}`~geoutils.Raster` or {class}`~geoutils.Vector` as a **reference to match** during the operation. A **reference {class}`~geoutils.Vector`** 
enforces a matching of `.bounds` and/or `.crs`, while a **reference {class}`~geoutils.Raster`** can also enforce a matching of `.res`, depending on the nature of the operation.


```{code-cell} ipython3
# Crop raster to vector's extent
rast.crop(vect)
```

```{code-cell} ipython3
# Compute proximity to vector on raster's grid
rast_proximity_to_vec = vect.proximity(rast)
```

All methods can be also be passed any number of georeferencing arguments such as `.shape` or `.res`, and will logically deduce others from the input, much 
as in [GDAL](https://gdal.org/)'s command line.


```{note}
Right now, the array `.data` of `rast` is still not loaded. Applying {func}`~geoutils.Raster.crop` does not yet require loading, 
and `rast`'s metadata is sufficient to provide a georeferenced grid for {func}`~geoutils.Vector.proximity`. The array will only be loaded when necessary.
```

Additionally, in GeoUtils, **methods that apply to the same georeferencing attributes have consistent naming**<sup>1</sup> across {class}`~geoutils.Raster` and {class}`~geoutils.Vector`. 

```{margin}
<sup>1</sup>The names of geospatial handling methods is largely based on [GDAL and OGR](https://gdal.org/)'s, with the notable exception of {func}`~geoutils.Vector.reproject` that better applies to vectors than `warp`.
```

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
   * - Proximity 
     - {func}`~geoutils.Raster.proximity`
     - {func}`~geoutils.Vector.proximity`
```

A {func}`~geoutils.Raster.reproject` involves a change in `.crs` or `.transform`, while a {func}`~geoutils.Raster.crop` only involves a change in `.bounds`. 
Using {func}`~geoutils.Raster.polygonize` allows to generate a {class}`~geoutils.Vector` from a {class}`~geoutils.Raster`, and the other way around for {func}`~geoutils.Vector.rasterize`. 


## Pythonic arithmetic and NumPy interface

All {class}`~geoutils.Raster` objects support Python arithmetic ({func}`+<operator.add>`, {func}`-<operator.sub>`, {func}`/<operator.truediv>`, {func}`//<operator.floordiv>`, {func}`*<operator.mul>`, 
{func}`**<operator.pow>`, {func}`%<operator.mod>`) with any other {class}`~geoutils.Raster`, {class}`~numpy.ndarray` or 
number. For other {class}`~geoutils.Raster`, the georeferencing must match, while only the shape for other {class}`~numpy.ndarray`.

```{code-cell} ipython3
# Add 1 to the raster array
rast += 1
```

Additionally, the class {class}`~geoutils.Raster` possesses a NumPy masked-array interface that allows to apply to it any [NumPy universal function](https://numpy.org/doc/stable/reference/ufuncs.html) and 
most other NumPy array functions, while logically casting `dtypes` and respecting `.nodata` values.

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

Masks also have simplified, overloaded {class}`~geoutils.Raster` methods due to their boolean `dtypes`. Using {func}`~geoutils.Raster.polygonize` with a 
{class}`~geoutils.Mask` is straightforward, for instance, to retrieve a {class}`~geoutils.Vector` of the area-of-interest:

```{code-cell} ipython3
# Polygonize areas where mask is True
vect_aoi = mask_aoi.polygonize()
```

## Plotting and saving geospatial data


Finally, GeoUtils includes basic plotting tools wrapping directly {func}`rasterio.plot.show` and {func}`geopandas.GeoDataFrame.plot` for {class}`~geoutils.Raster` and {class}`~geoutils.Vector`, respectively. 
The plotting function was renamed {func}`~geoutils.Raster.show` everywhere, for consistency. 

For saving, {func}`~geoutils.Raster.save` is used.

```{code-cell} ipython3
# Plot result
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.gca()
rast.show(ax=ax, cmap='Reds', cbar_title='Normalized infrared')
vect_aoi.ds.plot(ax=ax, fc='none', ec='k', lw=0.5)
```

```{admonition} Wrap-up
In a few lines, we:
 - **easily handled georeferencing** operations on rasters and vectors, 
 - performed numerical calculations **inherently respecting unvalid data**,
 - **naturally casted to a mask** from a logical operation on raster, and
 - **straightforwardly vectorized a mask** by harnessing overloaded subclass methods.

**Our result:** a vector of high infrared absorption indexes at least 200 meters away from glaciers 
near Everest, which likely corresponds to **perennial snowfields**.

For a **bonus** example on parsing satellite metadata and DEMs, continue below.

Otherwise, for more **hands-on** examples, explore GeoUtils' gallery of examples!
```

## **Bonus:** Parsing metadata with {class}`~geoutils.SatelliteImage`

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

There are many possible subclass to derive from a {class}`~geoutils.Raster`. Here's an **overview of current {class}`~geoutils.Raster` class inheritance**, which extends into 
[xDEM](https://xdem.readthedocs.io/en/latest/index.html) through the {class}`~xdem.DEM` class for analyzing digital elevation models: 

```{eval-rst}
.. inheritance-diagram:: geoutils.georaster.raster geoutils.georaster.satimg xdem.dem.DEM
    :top-classes: geoutils.georaster.raster.Raster
```
```{note}
The {class}`~xdem.DEM` class of [xDEM](https://xdem.readthedocs.io/en/latest/index.html) re-implements all methods of [gdalDEM](https://gdal.org/programs/gdaldem.html) 
(and more) to derive topographic attributes (hillshade, slope, aspect, etc), coded directly in Python for scalability and tested to yield the exact same 
results.
```