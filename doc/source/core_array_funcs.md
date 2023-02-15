(core-array-funcs)=

# Masked-array NumPy interface

NumPy possesses an [array interface](https://numpy.org/doc/stable/reference/arrays.interface.html) that allows to properly map their functions on objects 
that depend on a {class}`~numpy.ndarray`.

GeoUtils integrates this interface to work with any {class}`~geoutils.Raster` and their subclasses.

## Universal functions

The first category of NumPy functions that can be applied are [universal functions](https://numpy.org/doc/stable/reference/ufuncs.html) and 
most other NumPy array functions, while logically casting `dtypes` and respecting `.nodata` values.