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

GeoUtils is a Python package for **accessible**, **consistent** and **scalable** geospatial analysis.
::::

```{important}
:class: margin
GeoUtils ``v0.3`` is released with Xarray/GeoPandas accessors and full Dask support for rasters and point clouds (with benchmarking)! We are refining composability and vector scalability before a ``v1.0``.
```

GeoUtils is built on top of core geospatial packages (Rasterio, GeoPandas, PyProj) and numerical packages
(NumPy, SciPy, Numba) to provide **consistent higher-level functionalities for raster, vector and point
cloud objects** (such as reprojection, rasterization, polygonization, point interpolation, or gridding).

It is **computationally scalable** by implementing **lazy and chunked execution (Dask, Multiprocessing)** to all **raster, point cloud and vector** operations
and **provides accessors to naturally extend existing Python data-structures** (Xarray, Pandas).

GeoUtils is **tailored to perform quantitative analysis that implicitly understands the intricacies of geospatial data**
(nodata values, projection, pixel interpretation), through **an intuitive API to foster accessibility** (similar spirit as GDAL's new overhauled CLI).

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
{ref}`Learn more » <about-geoutils>`
:::

:::{grid-item-card} {material-regular}`data_exploration;2em` Quick start
:link: quick-start
:link-type: ref

Run a short example of the package functionalities.

+++
{ref}`Learn more » <quick-start>`
:::

:::{grid-item-card} {material-regular}`preview;2em` Features
:link: core-index
:link-type: ref

Dive into the full documentation.

+++
{ref}`Learn more » <core-index>`
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
scalability_index
data_object_index
referencing
transformations
proximity
filters
sampling
stats
```

```{toctree}
:caption: Resources
:maxdepth: 2

cheatsheet_osgeo
ecosystem
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
benchmarking_index
data
release_notes
```

```{toctree}
:caption: Project information
:maxdepth: 2

contributing
credits
```

# Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
