# GeoUtils
Set of tools to handle raster and vector data sets in Python.

![](https://readthedocs.org/projects/geoutils/badge/?version=latest)
[![build](https://github.com/GlacioHack/GeoUtils/actions/workflows/python-app.yml/badge.svg)](https://github.com/GlacioHack/GeoUtils/actions/workflows/python-app.yml)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/geoutils.svg)](https://anaconda.org/conda-forge/geoutils)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/geoutils.svg)](https://anaconda.org/conda-forge/geoutils)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/geoutils.svg)](https://anaconda.org/conda-forge/geoutils)
[![PyPI version](https://badge.fury.io/py/geoutils.svg)](https://badge.fury.io/py/geoutils)

This package offers Python classes and functions as well as command line tools to work with both geospatial raster and vector datasets. It is built upon rasterio and GeoPandas. In a single command it can import any geo-referenced dataset that is understood by these libraries, complete with all geo-referencing information, various helper functions and interface between vector/raster data.


## Installation

#### With conda (recommended)
```bash
conda install --channel conda-forge --strict-channel-priority geoutils
```
The `--strict-channel-priority` flag seems essential for Windows installs to function correctly, and is recommended for UNIX-based systems as well.

#### With pip

From PyPI:
```bash
pip install geoutils
```

Or from the repository tarball: make sure GDAL and PROJ are properly installed, then:
```bash
pip install https://github.com/GlacioHack/GeoUtils/tarball/main
```

## Documentation
See the full documentation at https://geoutils.readthedocs.io.


## Structure

GeoUtils are composed of three libraries:
- `georaster.py` to handle raster data set. In particular, a Raster class to load a raster file along with metadata.
- `geovector.py` to handle vector data set. In particular, a Vector class to load a raster file along with metadata.
- `projtools.py` with various tools around projections.


## How to contribute

You can find ways to improve the libraries in the [issues](https://github.com/GlacioHack/GeoUtils/issues) section. All contributions are welcome.

1. Fork the repository to your personal GitHub account, clone to your computer.
2. (Optional but preferred:) Make a feature branch.
3. Push to your feature branch.
4. When ready, submit a Pull Request from your feature branch to `GlacioHack/geoutils:master`.
5. The PR will be reviewed by at least one other person. Usually your PR will be merged via 'squash and merge'.

Direct pushing to the GlacioHack repository is not permitted.

A more detailed contribution instruction [can be found here](CONTRIBUTING.md).
