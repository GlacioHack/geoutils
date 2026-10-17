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

GeoUtils is a Python package for **scalable**, **composable** and **accessible** geospatial analysis.
::::

```{important}
:class: margin
GeoUtils ``v0.3`` introduces Xarray/GeoPandas accessors and finalizes Dask support for all raster and point cloud operations (with benchmarking)! GPU support is planned as we refine the API before a ``v1.0``.
```

GeoUtils is built on top of core geospatial packages (Rasterio, Shapely, PyProj) and numerical packages
(NumPy, SciPy, Numba) to provide **higher-level tools** for raster, vector and point
cloud objects, from transformations (e.g., reprojection, polygonization, point gridding) to
geostatistics (e.g., variography, random fields, kriging), and much more.

It adds **scalable** implementation respecting lazy and chunked execution (Dask, multiprocessing) for all raster
and point cloud operations and some vector operations, and is **composable** by exposing customizable methods
and backends that can be reused across operations. It also provides **accessors** to naturally extend
existing Python data structures (Xarray, GeoPandas).

Finally, it strives to make quantitative analysis **accessible** through an intuitive API, similar in spirit to
GDAL's overhauled CLI, while accounting implicitly for geospatial intricacies (nodata values, projections, pixel interpretation).

If you are looking to **port your GDAL or QGIS workflow in Python** while supporting large datasets, GeoUtils is made for you!

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
data_struct_index
referencing
transformations
proximity
filters
gapfill
sampling
stats
uncertainty
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
- {ref}`search`
