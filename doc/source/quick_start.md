(quick-start)=

# Quick start

The following presents how to quickly get started with GeoUtils, show-casing examples on different core aspects of the package.

For more details, refer to the {ref}`core-index`, {ref}`rasters-index` or {ref}`vectors-index` pages.

## The core {class}`~geoutils.Raster` and {class}`~geoutils.Vector` classes

In GeoUtils, geospatial handling is object-based and revolves around {class}`~geoutils.Raster` and {class}`~geoutils.Vector`.
These link to either **on-disk** or **in-memory** datasets, opened by instantiating the class.

```{literalinclude} code/index_example.py
:lines: 1-9
:language: python
```

A {class}`~geoutils.Raster` is a composition class with four main attributes: a {class}`~numpy.ma.MaskedArray` as `.data`, a {class}`~pyproj.crs.CRS` as `.crs`, 
an {class}`~affine.Affine` as `.transform`, and a {class}`float` or {class}`int` as `.nodata`. When a file exists on disk, {class}`~geoutils.Raster` is 
linked to a {class}`rasterio.DatasetReader` object for loading the metadata, and the array at the appropriate time.

A {class}`~geoutils.Vector` is a composition class with a single main attribute: a {class}`~geopandas.GeoDataFrame` as `.ds`, for which most methods are 
wrapped directly into {class}`~geoutils.Vector`. 

All other attributes are derivatives of those main attributes, or of the filename on disk. Attributes of {class}`~geoutils.Raster` and 
{class}`~geoutils.Vector` update with geospatial operations on themselves. 

## Geospatial handling and match-reference

Geospatial operations are based on class methods, such as {func}`geoutils.Raster.crop` or {func}`geoutils.Vector.proximity`. Nearly all of these methods can be 
passed solely another {class}`~geoutils.Raster` or {class}`~geoutils.Vector` as a **reference to match** during the operation. A **reference {class}`~geoutils.Vector`** 
enforces a matching of `.bounds` and/or `.crs`, while a **reference {class}`~geoutils.Raster`** can also enforce a matching of `.res`, depending on the nature of the operation.

```{literalinclude} code/index_example.py
:lines: 16-20
:language: python
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

All {class}`~geoutils.Raster` objects support Python arithmetic (+, -, /, //, *, ^, %) with any other {class}`~geoutils.Raster`, {class}`~numpy.ndarray` or 
number. For other {class}`~geoutils.Raster`, the georeferencing must match, while only the shape for other {class}`~numpy.ndarray`.

```{literalinclude} code/index_example.py
:lines: 22-23
:language: python
```

Additionally, the class {class}`~geoutils.Raster` possesses a NumPy masked-array interface that allows to apply to it any [NumPy universal function](https://numpy.org/doc/stable/reference/ufuncs.html) and 
most other NumPy array functions, while logically casting `dtypes` and respecting `.nodata` values.

```{literalinclude} code/index_example.py
:lines: 25-27
:language: python
```

## Casting to {class}`~geoutils.Mask`, indexing and overload

All {class}`~geoutils.Raster` classes also support Python logical operators (==, !=, >=, >, <=, <), or more complex NumPy logical functions. Those operations 
automatically casts them into a {class}`~geoutils.Mask`, a subclass of {class}`~geoutils.Raster`.


```{literalinclude} code/index_example.py
:lines: 29-30
:language: python
```

Masks can then be used for indexing a {class}`~geoutils.Raster`.

```{literalinclude} code/index_example.py
:lines: 32-33
:language: python
```

Masks also have simplified, overloaded {class}`~geoutils.Raster` methods due to their boolean `dtypes`. Using {func}`~geoutils.Raster.polygonize` with a 
{class}`~geoutils.Mask` is straightforward, for instance, to retrieve a {class}`~geoutils.Vector` of the area-of-interest:

```{literalinclude} code/index_example.py
:lines: 35-36
:language: python
```

## Plotting and saving geospatial data


Finally, GeoUtils includes basic plotting tools wrapping directly {func}`rasterio.plot.show` and {func}`geopandas.GeoDataFrame.plot` for {class}`~geoutils.Raster` and {class}`~geoutils.Vector`, respectively. 
The plotting function was renamed {func}`~geoutils.Raster.show` everywhere, for consistency. 

For saving, {func}`~geoutils.Raster.save` is used.

```{literalinclude} code/index_example.py
:lines: 38-47
:language: python
```

```{eval-rst}
.. plot:: code/index_example.py
    :caption: High infrared outside of glaciers at Mount Everest (perennial snowfields)
    :width: 90%
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

```{literalinclude} code/index_example.py
:lines: 11-14
:language: python
```

```{eval-rst}
.. program-output:: $PYTHON code/index_example.py
        :shell:
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