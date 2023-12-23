# GeoUtils: consistent geospatial analysis in Python.

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

**GeoUtils** is an open source project to develop a core Python package for geospatial analysis and foster inter-operability between other Python GIS packages.

It aims at **facilitating end-user geospatial analysis by revolving around consistent `Raster` and `Vector` objects** that effortlessly interface between
themselves. GeoUtils is founded on **implicit loading behaviour**, **robust numerical interfacing** and **convenient object-based methods** to easily perform
the most common higher-level tasks needed by geospatial users.

If you are looking for an accessible Python package to write the Python equivalent of your [GDAL](https://gdal.org/) command lines, or of your
[QGIS](https://www.qgis.org/en/site/) analysis pipeline **without a steep learning curve** on Python GIS syntax, GeoUtils is perfect for you! For more advanced
users, GeoUtils also aims at being efficient and scalable by supporting lazy loading and parallel computing (ongoing).

GeoUtils relies on [Rasterio](https://github.com/rasterio/rasterio), [GeoPandas](https://github.com/geopandas/geopandas) and [Pyproj](https://github.com/pyproj4/pyproj) for georeferenced
calculations, and on [NumPy](https://github.com/numpy/numpy) and [Xarray](https://github.com/pydata/xarray) for numerical analysis. It allows easy access to
the functionalities of these packages through interfacing or composition, and quick inter-operability through object conversion.

## Documentation

For a quick start, full feature description or search through the API, see GeoUtils' documentation at: https://geoutils.readthedocs.io.

## Installation

```bash
mamba install -c conda-forge geoutils
```

See [mamba's documentation](https://mamba.readthedocs.io/en/latest/) to install `mamba`, which will solve your environment much faster than `conda`.

## Start contributing

1. Fork the repository, make a feature branch and push changes.
2. When ready, submit a pull request from the feature branch of your fork to `GlacioHack/geoutils:main`.
3. The PR will be reviewed by at least one maintainer, discussed, then merged.

More info on [our contributing page](CONTRIBUTING.md).
