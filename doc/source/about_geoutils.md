(about-geoutils)=

# About GeoUtils

## What is GeoUtils?

GeoUtils<sup>1</sup> is an open source **[Python](https://www.python.org/) package for geospatial analysis**, aiming to improve
**scalability and interoperability** with Python GIS packages.

```{margin}
<sup>1</sup>With name standing for *Geospatial Utilities*.
```

GeoUtils **can be seen primarily as an extension of [Xarray](https://docs.xarray.dev/en/stable/) and
[GeoPandas](https://geopandas.org/en/stable/docs.html)**, adding **higher-level analysis tools** at the interface of
rasters, vectors and point clouds, from operations such as transformations (e.g., reprojection, point gridding,
polygonization) to more advanced statistics (e.g., grouped stats, spatial correlation estimation, random fields)<sup>2</sup>.

```{margin}
<sup>2</sup> Explore the {ref}`full feature list <feature-overview>`.
```

GeoUtils adds **lazy, chunked execution** with [Dask](https://docs.dask.org/en/stable/) for all raster and point cloud
operations and some vector operations<sup>3</sup>. Operations rely directly on the Python statistical and geospatial stack (e.g.,
[Rasterio](https://rasterio.readthedocs.io/en/latest/), [PyProj](https://pyproj4.github.io/pyproj/stable/index.html),
[SciPy](https://docs.scipy.org/doc/scipy/), [Numba](https://numba.readthedocs.io/en/stable/),
[GSTools](https://geostat-framework.readthedocs.io/projects/gstools/en/stable/),
[SciKit-GStat](https://scikit-gstat.readthedocs.io/en/latest/), [GPyTorch](https://docs.gpytorch.ai/en/stable/)),
with **composable** methods and backends that can be easily
customized while GeoUtils handles georeferencing and chunked execution.
Finally, GeoUtils strives to make these tools **accessible** through a shared interface across rasters, point clouds and
vectors, with a structure following [the principle of least astonishment](https://en.wikipedia.org/wiki/Principle_of_least_astonishment) for software development.

```{margin}
<sup>3</sup> Check {ref}`scalability-support` with Dask, and **visualize chunked implementations** through their
{ref}`logic diagrams <scalability-logic>`.
```

For example, a kriging interpolator can be reused for raster resampling, gap-filling or point gridding, and a weighted mean
reducer for resampling, filtering or zonal statistics<sup>4</sup>. With kriging, **we don't reinvent the underlying mathematics**,
instead relying on existing geostatistical packages where relevant, and adding only specific logic to make the operations
scalable on large data (out-of-memory pair sampling and reduction, covariance-aware chunking).

```{margin}
<sup>4</sup> See {ref}`core-operators` for reusable interpolators and reducers.
```

GeoUtils is designed for all Earth and planetary observation science, and generally **most useful for quantitative analysis of large datasets**.

## Who is behind GeoUtils?

GeoUtils was created by a group of researchers with expertise in geostatistics, GIS, and remote sensing of the cryosphere.
Nowadays, its development is **jointly led by researchers in geospatial analysis** (including funding from NASA and SNSF) **and
engineers from CNES** (French Space Agency).

Most contributors and users are scientists or industrials working in **various fields of Earth observation**<sup>5</sup>.

```{margin}
<sup>5</sup> More details about the people behind GeoUtils, funding sources, and the package's objectives can be found on the **{ref}`credits` pages**.
```

::::{grid}
:reverse:

:::{grid-item}
:columns: 4
:child-align: center

```{image} ./_static/nasa_logo.svg
    :width: 200px
    :class: dark-light
```

:::

:::{grid-item}
:columns: 4
:child-align: center

```{image} ./_static/snsf_logo.svg
    :width: 220px
    :class: only-light
```

```{image} ./_static/snsf_logo_dark.svg
    :width: 220px
    :class: only-dark
```

:::

:::{grid-item}
:columns: 4
:child-align: center

```{image} ./_static/cnes_logo.svg
    :width: 200px
    :class: only-light
```

```{image} ./_static/cnes_logo_dark.svg
    :width: 200px
    :class: only-dark
```

:::


::::

## Things GeoUtils does for you under the hood

GeoUtils does not only add scalability to existing features, but also **adds modularity or optimization under-the-hood**
for more efficient geospatial analysis.

For example:

We **add customizable NaN handling on top of existing filters and interpolators**, to facilitate analysis with missing data<sup>6</sup>.
```{margin}
<sup>6</sup> Choose a **{ref}`nodata propagation rule <core-nodata>`**.
```

We **interpolate large rasters much faster** than Xarray's {meth}`~xarray.DataArray.interp` (based on SciPy's rectilinear interpolation)
by simply relying on better-adapted equal-grid interpolation from {func}`~scipy.ndimage.map_coordinates`.<sup>7</sup>

```{margin}
<sup>7</sup> Compare **{ref}`resampling methods and backends <transformations-resampling>`**.
```

We **support varied statistics on datasets larger than memory**, building on Dask/Flox reductions<sup>8</sup>, but
adding more direct support for **subsampling**, **weights** or **zonal statistics from vectors**, including fractional
geometry-grid intersections.

```{margin}
<sup>8</sup> See **{ref}`stats-grouped`** for grouping and chunked reductions.
```

We **facilitate estimating spatial correlations** by efficiently sampling pairs out-of-memory from large rasters or point clouds while
equally representing all distances (as random samples on large data tend to rarely fall on two neighbouring points!). We also
efficiently compute **variography estimators on larger-than-memory spatial pairs**, building on Dask/Flox as for the statistics above.
During all of this, we reuse [SciKit-GStat](https://scikit-gstat.readthedocs.io/en/latest/) estimators and models to add the smallest layer necessary for scalability,
and provide conversion to other covariance models (GSTools, GPyTorch) to facilitate interoperability.<sup>9</sup>.

```{margin}
<sup>9</sup> See **{ref}`sampling-pairs`** and **{ref}`spatial correlation estimation <uncertainty-correlation>`**.
```
