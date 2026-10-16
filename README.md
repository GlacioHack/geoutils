# GeoUtils: scalable and composable geospatial analysis in Python.

![](https://readthedocs.org/projects/geoutils/badge/?version=latest)
[![build](https://github.com/GlacioHack/geoutils/actions/workflows/python-tests.yml/badge.svg)](https://github.com/GlacioHack/GeoUtils/actions/workflows/python-tests.yml)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/geoutils.svg)](https://anaconda.org/conda-forge/geoutils)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/geoutils.svg)](https://anaconda.org/conda-forge/geoutils)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/geoutils.svg)](https://anaconda.org/conda-forge/geoutils)
[![PyPI version](https://badge.fury.io/py/geoutils.svg)](https://badge.fury.io/py/geoutils)
[![Coverage Status](https://coveralls.io/repos/github/GlacioHack/geoutils/badge.svg?branch=main)](https://coveralls.io/github/GlacioHack/geoutils?branch=main)

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/GlacioHack/geoutils/main)
[![Pre-Commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Formatted with black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

**GeoUtils** is an open source Python package for geospatial analysis, aiming to improve scalability and interoperability with Python GIS packages.

GeoUtils **can be seen primarily as an extension of [Xarray](https://github.com/pydata/xarray) and [GeoPandas](https://github.com/geopandas/geopandas)**, adding **higher-level analysis tools** at the interface of rasters, vectors and point clouds, from operations such as transformations (e.g., reprojection, point gridding, polygonization) to more advanced statistics (e.g., grouped stats, spatial correlation estimation, random fields). See the [full feature list](#features-list) further below.

Primarily, GeoUtils adds **scalable implementations** to support chunked processing of datasets larger than memory, including lazy [Dask](https://docs.dask.org/en/stable/) support for most raster and point cloud operations and some vector operations.
Operations rely directly on the Python statistical and geospatial stack (e.g., [Rasterio](https://github.com/rasterio/rasterio), [Pyproj](https://github.com/pyproj4/pyproj), [NumPy](https://github.com/numpy/numpy), [SciPy](https://github.com/scipy/scipy), [Numba](https://numba.readthedocs.io/en/stable/)), with **composable** methods and backends that can be easily customized while GeoUtils handles georeferencing and chunked execution.
GeoUtils strives to make these tools **accessible** through a shared interface across rasters, point clouds and vectors, with naming consistent with the [recent GDAL CLI overhaul](https://gdal.org/en/stable/programs/gdal.html).

For example, a kriging interpolator can be reused for raster resampling or point gridding, and a weighted mean reducer for resampling, filtering or zonal statistics.
As another example, we add efficient Dask pair sampling and estimator reduction to support scalable estimation of spatial correlation, while reusing estimators and models from [SciKit-GStat](https://scikit-gstat.readthedocs.io/en/latest/) to not reinvent the underlying mathematics.
These models can then be applied through [GSTools](https://geostat-framework.readthedocs.io/projects/gstools/en/stable/) or [GPyTorch](https://docs.gpytorch.ai/en/stable/) for kriging and random fields, here again only adding the logic enabling GIS-aware, scalable execution.

Accessors expose these operations directly on native Xarray and GeoPandas data structures:

| Data type | Native data structure | GeoUtils accessor |
|---|---|---|
| Raster | Xarray `DataArray` or `Dataset` | `.rst` |
| Vector | GeoPandas `GeoDataFrame` | `.vct` |
| Point cloud | Xarray `DataArray` or `Dataset`, or GeoPandas `GeoDataFrame` | `.pc` |

GeoUtils also implements its own **`Raster`, `Vector` and `PointCloud` objects for Python beginners**, so users can manipulate geospatial data
without first learning labeled arrays or dataframes. Internally, **operations are implemented independently of the data structures**, allowing to easily
add accessors to emerging data structures such as [GeoPolars](https://geopolars.org/latest/).

If you are looking for an accessible Python package to write the Python equivalent of your [GDAL](https://gdal.org/) command lines, or of your
[QGIS](https://www.qgis.org/en/site/) analysis pipeline **without a steep learning curve** on Python GIS syntax, GeoUtils is perfect for you!

The **core installation is lightweight** for users already working with Xarray and GeoPandas, relying on the standard geospatial and numerical stack.
**Most feature-specific dependencies are optional**, including Dask for chunked execution, Matplotlib for plotting, LasPy and PyArrow for point cloud
formats, [SciKit-GStat](https://scikit-gstat.readthedocs.io/en/latest/) for variograms, and [GSTools](https://geostat-framework.readthedocs.io/projects/gstools/en/stable/) or [GPyTorch](https://docs.gpytorch.ai/en/stable/) for kriging and random field simulation.

## Features list

GeoUtils supports the features listed below, primarily extending them from existing tools (SciPy, Numba, SciKit-GStat, etc) to be **scalable**, **composable** and more **accessible** in the context of geospatial analysis:

- **Referencing:** manipulate coordinate systems, bounds, footprints, grid transforms and pixel interpretation.
- **Transformations:** crop, clip, reproject, rasterize, polygonize, resample at points, grid point clouds, and create vector geometry masks.
- **Proximity:** distance calculations and buffering without overlap.
- **Filters:** filter rasters and point clouds using shared methods that handle missing data.
- **Gap-filling:** estimate missing values from nearby valid observations.
- **Sampling:** subsampling, co-sampling and pair sampling.
- **Statistics:** global, grouped and zonal statistics, variograms and kriging.
- **Uncertainty:** error structure estimation, spatial uncertainty propagation and spatially correlated random fields.

## Quick start

Below we **compare an elevation raster with LiDAR point observations using kriging**, both opened in chunks to perform all following operations out-of-memory.

```python
import geoutils as gu

# Open big raster and point files as lazy Xarray arrays in chunks
ds = gu.open_raster("dem.tif", chunks={"x": 512, "y": 512})
ds_lidar = gu.open_pointcloud("lidar.laz", data_name="Z", chunks=10_000)

# Estimate spatial correlation from sampled pairs without loading the whole raster
variogram = ds.rst.variogram(
    n_pairs=20_000, estimator="matheron", model="spherical", random_state=42
)

# Krige raster elevations at the point locations using nearby 5 x 5 cells,
# without loading any of the two inputs
ds_point = ds.rst.resample_at_points(
    ds_lidar,  # Can simply pass point input to "match" coordinates where we resample
    resampling="kriging",
    backend="gstools",
    resampling_kwargs={"model": variogram, "neighborhood": gu.operators.GridNeighbours(size=5)},
)

# Subtract observed elevations lazily, then compute bias, spread and RMSE, also all in chunks
ds_diff = ds_point - ds_lidar
print(ds_diff.pc.stats(["mean", "std", "rmse"]))
```

## Documentation

For a quick start, full feature description or search through the API, see GeoUtils' documentation at: https://geoutils.readthedocs.io.

## Installation

Install with `conda`/`mamba`:

```bash
mamba install -c conda-forge geoutils
```

Or `pip`:

```bash
pip install geoutils
```

To run the kriging example above, include optional dependencies directly:

```bash
pip install "geoutils[opt]"
```

See the [installation guide](https://geoutils.readthedocs.io/en/latest/how_to_install.html) for more details.

## Start contributing

We welcome new contributions!

A short guide:

1. Fork the repository, make a feature branch and push changes.
2. When ready, submit a pull request from the feature branch of your fork to `GlacioHack/geoutils:main`.
3. The PR will be reviewed by at least one maintainer, discussed, then merged.

More info on [our contributing page](CONTRIBUTING.md).
