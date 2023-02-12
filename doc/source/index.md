% GeoUtils documentation master file, created by
% sphinx-quickstart on Fri Nov 13 17:43:16 2020.
% You can adapt this file completely to your liking, but it should at least
% contain the root 'toctree' directive.

# Welcome to GeoUtils' documentation!

GeoUtils aims to make the handling and analysis of georeferenced raster and vector data intuitive, robust, and easy-of-use. GeoUtils is built upon `rasterio`, 
`geopandas` and `pyproj`, and facilitates geospatial operations between rasters and vectors, and the analysis of rasters by interfacing with `numpy` arrays.


```{important}
:class: margin
GeoUtils is in early stages of development and its features might evolve rapidly. Note the version you are working on for
**reproducibility**!
We are working on making features fully consistent for the first long-term release ``v0.1`` (likely sometime in 2023).
```


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
    
auto_examples/index.rst 
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
