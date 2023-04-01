# GeoUtils

Handling and analysis of georeferenced rasters and vectors in Python.

![](https://readthedocs.org/projects/geoutils/badge/?version=latest)
[![build](https://github.com/GlacioHack/geoutils/actions/workflows/python-app.yml/badge.svg)](https://github.com/GlacioHack/GeoUtils/actions/workflows/python-app.yml)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/geoutils.svg)](https://anaconda.org/conda-forge/geoutils)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/geoutils.svg)](https://anaconda.org/conda-forge/geoutils)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/geoutils.svg)](https://anaconda.org/conda-forge/geoutils)
[![PyPI version](https://badge.fury.io/py/geoutils.svg)](https://badge.fury.io/py/geoutils)
[![Coverage Status](https://coveralls.io/repos/github/GlacioHack/geoutils/badge.svg?branch=main)](https://coveralls.io/github/GlacioHack/geoutils?branch=main)

[![Pre-Commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Formatted with black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![isort Status]](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

GeoUtils is community effort to develop a core Python package for geospatial analysis and to foster inter-operability between Python GIS packages.
It aims at facilitating end-user geospatial tasks by revolving around consistent `Raster` and `Vector` objects that interface easily, have implicit
loading behaviour, robust numerical interfacing and convenient single-line methods for all the most common higher-level methods used in the geospatial
community.

GeoUtils relies on [Rasterio](https://github.com/rasterio/rasterio), [GeoPandas](https://github.com/geopandas/geopandas) and [Pyproj](https://github.com/pyproj4/pyproj) for georeferenced
calculations, and on [NumPy](https://github.com/numpy/numpy) and [Xarray](https://github.com/pydata/xarray) for numerical analysis.

## Documentation

For a quick start, gallery examples, a full feature description or a search through the API, see GeoUtils' documentation at: https://geoutils.readthedocs.io.

## Installation

```bash
mamba install -c conda-forge geoutils
```

## Start contributing

1. Fork the repository, make a feature branches and push changes.
2. When ready, submit a pull request from the feature branch of your fork to `GlacioHack/geoutils:master`.
3. The PR will be reviewed by at least one maintainer, discussed, then merged.

More info on [our contributing page](CONTRIBUTING.md).
