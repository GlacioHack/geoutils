(about-geoutils)=

# About GeoUtils

## What is GeoUtils?

GeoUtils<sup>1</sup> is a [Python](https://www.python.org/) package for the manipulation and analysis of georeferenced data, developed with the objective of 
making geospatial analysis intuitive, accessible and robust. It is designed for all Earth and planetary observation science.

```{margin}
<sup>1</sup>With name standing for *Geospatial Utilities*.
```

```{important}
GeoUtils is in early stages of development and its features might evolve rapidly. Note the version you are working on for
**reproducibility**!
We are working on making features fully consistent for the first long-term release ``v0.1`` (likely sometime in 2023).
```

## Why use GeoUtils?

GeoUtils is built on top of the packages [Rasterio](https://rasterio.readthedocs.io/en/latest/), [GeoPandas](https://geopandas.org/en/stable/docs.html) 
and [PyProj](https://pyproj4.github.io/pyproj/stable/index.html) for georeferenced operations, and relies on [NumPy](https://numpy.org/doc/stable/), 
[SciPy](https://docs.scipy.org/doc/scipy/) and [Xarray](https://docs.xarray.dev/en/stable/) for scientific computing to provide:
- A **common and consistent framework** for rasters and vectors handling and analysis,
- A structure following the **principal of least knowledge**<sup>2</sup> to foster accessibility,
- A **pythonic arithmetic** and **NumPy interfacing** for intuitive use.

```{margin}
<sup>2</sup>Or the [Law of Demeter](https://en.wikipedia.org/wiki/Law_of_Demeter) for software development.
```

In particular, GeoUtils:
- Rarely requires more than **single-line operations** due to its object-based structure,
- Allows for **match-reference operations** to facilitate geospatial handling,
- Re-implements **several of [GDAL](https://gdal.org/)'s missing features** (Proximity, DEM, Calc, etc),
- Naturally handles **different `dtypes` and `nodata`** values through its NumPy masked-array interface.


```{note}
More on these core features of GeoUtils in the {ref}`quick-start`, or {ref}`core-index` for details.
```