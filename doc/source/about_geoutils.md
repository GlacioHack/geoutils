(about-geoutils)=

# About GeoUtils

## What is GeoUtils?

GeoUtils is a [Python](https://www.python.org/) package for the manipulation and analysis of georeferenced data, developed with the objective of 
making geospatial analysis intuitive, accessible and robust. It is designed for all Earth and planetary observation science.

GeoUtils is built on top of the geospatial packages [Rasterio](https://rasterio.readthedocs.io/en/latest/), [GeoPandas](https://geopandas.org/en/stable/docs.
html) and [PyProj](https://pyproj4.github.io/pyproj/stable/index.html) to provide:
- A **common consistent interface** between georeferenced rasters and vectors for robustness of geospatial handling,
- A geospatial framework following the **principal of least knowledge** for ease-of-use and accessibility,

Additionally




## Mission

```{epigraph}
The core mission of GeoUtils is to be **easy-of-use**, **robust**, **reproducible** and **fully open**.

Additionally, GeoUtils aims to be **efficient**, **scalable** and **state-of-the-art**.
```

```{important}
:class: margin
GeoUtils is in early stages of development and its features might evolve rapidly. Note the version you are working on for
**reproducibility**!
We are working on making features fully consistent for the first long-term release ``v0.1`` (likely sometime in 2023).
```

In details, those mean:

- **Ease-of-use:** all basic operations or methods only require a few lines of code to be performed;

- **Robustness:** all methods are tested within our continuous integration test-suite, to enforce that they always perform as expected;

- **Reproducibility:** all code is version-controlled and release-based, to ensure consistency of dependent packages and works;

- **Open-source:** all code is accessible and re-usable to anyone in the community, for transparency and open governance.

```{note}
:class: margin
Additional mission points, in particular **scalability**, are partly developed but not a priority until our first long-term release ``v0.1`` is reached.
```

And, additionally:

- **Efficiency**: all methods should be optimized at the lower-level, to function with the highest performance offered by Python packages;

- **Scalability**: all methods should support both lazy processing and distributed parallelized processing, to work with high-resolution data on local machines as well as on HPCs;

- **State-of-the-art**: all methods should be at the cutting edge of remote sensing science, to provide users with the most reliable and up-to-date tools.
