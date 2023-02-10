(quick-start)=

# Quick start

Our functionalities are mostly based on [rasterio](https://rasterio.readthedocs.io) and [GeoPandas](https://geopandas.org/), but here are some of the
additional benefits provided by GeoUtils:

- for raster objects, georeferences and data are stored into a single object of the class `Raster`, making it easier to modify the data in-place: reprojection, cropping, additions/subtractions are all (or will be...) on-line operations!
- the interactions between raster and vectors, such as rasterizing, clipping or cropping are made easier thanks to the class `Vector`.
