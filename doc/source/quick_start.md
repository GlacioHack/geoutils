(quick-start)=

# Quick start

## The core `Raster` and `Vector` classes

In GeoUtils, geospatial handling is object-based and revolves around {class}`~geoutils.Raster` and {class}`~geoutils.Vector`.
These link to either on-disk or in-memory datasets, opened by instantiating the class.

```{literalinclude} code/index_example.py
:lines: 1-9
:language: python
```

## Class attributes and match-reference methods

Attributes of {class}`~geoutils.Raster` and {class}`~geoutils.Vector` update with operations. These operations are generally based on class methods, for 
example {func}`~geoutils.Raster.crop` or {func}`~geoutils.Vector.proximity`. Most of these functions can be passed solely another {class}`~geoutils.Raster` or 
{class}`~geoutils.Vector` as a reference to match, or to utilize during the operation.

```{literalinclude} code/index_example.py
:lines: 11-15
:language: python
```

```{note}
Right now, the array `.data` of `rast` is still not loaded. Applying {func}`~geoutils.Raster.crop` does not yet require loading, 
and the metadata is sufficient to provide a georeferenced grid for {func}`~geoutils.Vector.proximity`. The array data will only be loaded when necessary.
```

## Pythonic arithmetic and NumPy interface

All {class}`~geoutils.Raster` support Python arithmetic (+, -, /, //, *, %) with any other {class}`~geoutils.Raster`, or {class}`~numpy.ndarray` or number
(in case the georeferencing or shape is the same for all objects).

```{literalinclude} code/index_example.py
:lines: 17-18
:language: python
```

Additionally, the class {class}`~geoutils.Raster` possesses a NumPy masked-array interface that allows to apply to it any [NumPy universal function](https://numpy.org/doc/stable/reference/ufuncs.html) and 
most other NumPy array functions, while logically casting `dtypes` and respecting `.nodata` values.

```{literalinclude} code/index_example.py
:lines: 20-22
:language: python
```

## Casting to `Mask`, indexing and overloading

All {class}`~geoutils.Raster` also support Python logical operators (==, !=, >=, >, <=, <), or more complex NumPy logical functions. Those operations 
automatically casts them into a {class}`~geoutils.Mask`, a subclass of {class}`~geoutils.Raster`.


```{literalinclude} code/index_example.py
:lines: 24-25
:language: python
```

Masks can then be used for indexing a {class}`~geoutils.Raster`.

```{literalinclude} code/index_example.py
:lines: 27-28
:language: python
```

Masks also have simplified, overloaded {class}`~geoutils.Raster` methods due to their boolean `dtypes`. Polygonizing a `Mask` is straightforward, for instance,
to retrieve a {class}`~geoutils.Vector` of the area-of-interest identified by the mask:

```{literalinclude} code/index_example.py
:lines: 30-31
:language: python
```

## Loading image metadata using `SatelliteImage`

The {class}`~geoutils.Raster` class is also subclassed by {class}`~geoutils.SatelliteImage`, which tentatively parses metadata recognized from the filename 
or auxiliary files.

```{literalinclude} code/index_example.py
:lines: 33-35
:language: python
```

```{eval-rst}
.. program-output:: $PYTHON code/index_example.py
        :shell:
```