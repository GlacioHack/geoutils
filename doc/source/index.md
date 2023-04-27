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

GeoUtils is a Python package for **accessible**, **efficient** and **reliable** geospatial analysis.
::::

```{tip}
:class: margin
**Run any page of this documentation interactively** by clicking the top launch button!

Or **start your own test notebook**: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/GlacioHack/geoutils/main).
```

**Accessible** owing to its convenient object-based structure, intuitive match-reference operations and familiar geospatial dependencies
([Rasterio](https://rasterio.readthedocs.io/en/latest/), [Rioxarray](https://corteva.github.io/rioxarray/stable/),
[GeoPandas](https://geopandas.org/en/stable/docs.html), [PyProj](https://pyproj4.github.io/pyproj/stable/index.html)).

**Efficient** owing to its implicit lazy loading functionalities, logical integration with pythonic operators and array interfacing
([NumPy](https://numpy.org/doc/stable/), [SciPy](https://docs.scipy.org/doc/scipy/) and [Xarray](https://docs.xarray.dev/en/stable/)).

**Reliable** owing to its consistent higher-level operations respecting geospatial intricacies such as nodata values and pixel interpretation, ensured by
its testing suite and type checking ([Pytest](https://docs.pytest.org/en/7.2.x/), [Mypy](https://mypy-lang.org/)).

----------------

# Where to start?

```{important}
:class: margin
GeoUtils is in early stages of development and its features might evolve rapidly. Note the version you are working on for
**reproducibility**!
We are working on making features fully consistent for the first long-term release ``v0.1`` (likely sometime in 2023).
```

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
```

```{toctree}
:caption: Features
:maxdepth: 2

core_index
rasters_index
vectors_index
proj_tools
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
background
```

# Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
