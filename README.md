# GeoUtils
Set of tools to handle raster and vector data sets in Python.

This package offers Python classes and functions as well as command line tools to work with both geospatial raster and vector datasets. It is built upon rasterio and GeoPandas. In a single command it can import any geo-referenced dataset that is understood by these libraries, complete with all geo-referencing information, various helper functions and interface between vector/raster data.


## Installation ##

* Create environment and install dependencies

`conda env create -f environment.yml`

* Activate environment

`conda activate geoutils`

* Install this package

`pip install -e .` or `python setup.py install`

* Check that everything is working by running the tests

`pytest -rA`



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


### Documentation - please read ! ###
In the interest of keeping the documentation simple, please write all docstring in reStructuredText (https://docutils.sourceforge.io/rst.html) format - eventually, we will try to set up auto-documentation using sphinx and readthedocs, and this will help in that task.

### Testing - again please read!
These tools are only valuable if we can rely on them to perform exactly as we expect. So, we need testing. Please create tests for every function that you make, as much as you are able. Guidance/examples here for the moment: 
- https://github.com/GeoUtils/georaster/blob/master/test/test_georaster.py
- https://github.com/corteva/rioxarray/blob/master/test/integration/test_integration__io.py

