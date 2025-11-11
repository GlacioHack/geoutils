---
title: GeoUtils
---

::::{grid}
:reverse:
:gutter: 2 1 1 1
:margin: 4 4 1 1

:::{grid-item}
:columns: 4

```{image} ./_static/logo_only.png
    :width: 300px
    :class: dark-light
```
:::

:::{grid-item}
:columns: 8
:class: sd-fs-3
:child-align: center

GeoUtils is a Python package for **accessible** and **consistent** geospatial analysis.
::::

```{important}
:class: margin
GeoUtils ``v0.2`` is released with more consistent point cloud support! We are working on Xarray and GeoPandas accessors for all data objects, as well as other scalability features.
```

GeoUtils is built on top of core geospatial packages (Rasterio, GeoPandas, PyProj) and numerical packages
(NumPy, Xarray, SciPy) to provide **consistent higher-level functionalities at the interface of raster, vector and point
cloud objects** (such as match-reference reprojection, point interpolation or gridding).

It is **tailored to perform quantitative analysis that implicitly understands the intricacies of geospatial data**
(nodata values, projection, pixel interpretation), through **an intuitive object-based API to foster accessibility**,
and strives **to be computationally scalable** (Dask support in development for future Xarray accessor).

If you are looking to **port your GDAL or QGIS workflow in Python**, GeoUtils is made for you!

----------------

# Where to start?

::::{grid} 1 2 2 3
:gutter: 1 1 1 2

:::{grid-item-card} {material-regular}`edit_note;2em` About GeoUtils
:link: about-geoutils
:link-type: ref

Learn more about why we developed GeoUtils.

+++
[Learn more »](about_geoutils)
:::

:::{grid-item-card} {material-regular}`data_exploration;2em` Quick start
:link: quick-start
:link-type: ref

Run a short example of the package functionalities.

+++
[Learn more »](quick_start)
:::

:::{grid-item-card} {material-regular}`preview;2em` Features
:link: core-index
:link-type: ref

Dive into the full documentation.

+++
[Learn more »](core_index)
:::

::::

Prefer to **grasp GeoUtils' core concepts by comparing with other Python packages**? Read through a short **{ref}`side-by-side code comparison with Rasterio and GeoPandas<comparison-rasterio-geopandas>`**.

Looking to **learn a specific feature by running an example**? Jump straight into our **example galleries on {ref}`examples-io`, {ref}`examples-handling` and {ref}`examples-analysis`**.


```{seealso}
If you are DEM-enthusiastic, **[check-out our sister package xDEM](https://xdem.readthedocs.io/) for digital elevation models.**
```
----------------

# Table of contents

```{toctree}
:caption: Getting started
:maxdepth: 2

about_geoutils
how_to_install
quick_start
feature_overview
```

```{toctree}
:caption: Features
:maxdepth: 2

core_index
data_object_index
georeferencing
geotransformations
raster_vector_point
distance_ops
stats
```

```{toctree}
:caption: Examples
:maxdepth: 2

io_examples/index
handling_examples/index
analysis_examples/index
```

```{toctree}
:caption: Reference
:maxdepth: 2

api
cli
config
release_notes
```

```{toctree}
:caption: Project information
:maxdepth: 2

credits
```

# Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
