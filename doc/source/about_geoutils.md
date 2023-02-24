(about-geoutils)=

# About GeoUtils

## What is GeoUtils?

GeoUtils<sup>1</sup> is a **[Python](https://www.python.org/) package for the handling and analysis of georeferenced data**, developed with the objective of 
making such analysis accessible, efficient and reliable. 

```{margin}
<sup>1</sup>With name standing for *Geospatial Utilities*.
```

In a few words, GeoUtils can be described as a **convenience wrapper package for end-users** focusing on geospatial analysis. It allows to write shorter 
code through consistent higher-level operations, implicit object behaviour and numerical interfacing. In addition, GeoUtils adds **analysis-oriented 
functions** that require many steps to perform with other packages, and which are robustly tested.

GeoUtils is designed for all Earth and planetary observation science. However, it is generally **most useful for remote sensing and Earth's surface 
applications** that rely on moderate- to high-resolution georeferenced data. All applications that, for analysis, require robust reprojections, re-gridding, 
point interpolation, and other types of fine-grid analysis with millions of pixels.


```{important}
GeoUtils is in early stages of development and its features might evolve rapidly. Note the version you are working on for
**reproducibility**!
We are working on making features fully consistent for the first long-term release ``v0.1`` (likely sometime in 2023).
```

## Why use GeoUtils?

GeoUtils is built on top of [Rasterio](https://rasterio.readthedocs.io/en/latest/), [GeoPandas](https://geopandas.org/en/stable/docs.html) 
and [PyProj](https://pyproj4.github.io/pyproj/stable/index.html) for georeferenced operations, and relies on [NumPy](https://numpy.org/doc/stable/), 
[SciPy](https://docs.scipy.org/doc/scipy/) and [Xarray](https://docs.xarray.dev/en/stable/) for scientific computing to provide:
- A **common and consistent framework** for efficient rasters and vectors handling,
- A structure following the **principal of least knowledge**<sup>2</sup> to foster accessibility,
- A **pythonic arithmetic** and **NumPy interfacing** for robust numerical computing and intuitive analysis.

```{margin}
<sup>2</sup>Or the [Law of Demeter](https://en.wikipedia.org/wiki/Law_of_Demeter) for software development.
```

In particular, GeoUtils:
- Rarely requires more than **single-line operations** thanks to its object-based structure,
- Strives to rely on **lazy-operations** under-the-hood to avoid unnecessary data loading,
- Allows for **match-reference operations** to facilitate geospatial handling,
- Re-implements **several of [GDAL](https://gdal.org/)'s missing features** (Proximity, DEM, Calc, etc),
- Naturally handles **different `dtypes` and `nodata`** values through its NumPy masked-array interface.


```{note}
More on these core features of GeoUtils in the {ref}`quick-start`, or {ref}`core-index` for details.
```

## Why the need for GeoUtils?

Recent community efforts have improved open-source geospatial analysis in Python, allowing to **move away from the low-level functions and 
complexity of [GDAL and OGR](https://gdal.org/)**'s Python bindings for raster and vector handling. Those efforts include in particular [Rasterio](https://rasterio.readthedocs.io/en/latest/) and [GeoPandas](https://geopandas.org/en/stable/docs.html).

However, these new packages still maintain a relatively low-level API to serve all types of geospatial informatics users, **slowing down end-users focusing 
on data analysis**. As a result, interfacing between vector data and raster data is delicate and simple higher-level operation (such as 
reprojection to match a reference) are not always computed consistently in the community.

Additionally, [Rasterio](https://rasterio.readthedocs.io/en/latest/) focuses mostly on reading, projecting and writing, and thus **requires array extraction 
and re-encapsulation** either before, during or after any numerical computation to interface with other georeferencing packages. 

Finally, **many common geospatial analysis tools are generally unavailable** in existing packages (e.g., boolean-masking from vectors, 
[proximity](https://gdal.org/programs/gdal_proximity.html) estimation, metric buffering) as they rely on a combination of lower-level operations. 

```{admonition} Conclusion
Having higher-level geospatial tools implemented in a **consistent** manner and tested for **robustness** is essential for the wider geospatial community.
```