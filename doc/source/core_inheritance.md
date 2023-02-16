---
file_format: mystnb
kernelspec:
  name: geoutils
---
(core-inheritance)=
# Inheritance to geo-images and beyond

Inheritance is practical to naturally pass down parent methods and attributes to child classes. 
In the case of {class}`Rasters<geoutils.Raster>`, many types of geospatial data exist with their own peculiarities, additional attributes, while 
remaining a {class}`Rasters<geoutils.Raster>`.

## Overview of {class}`~geoutils.Raster` inheritance


Below is a diagram showing current {class}`~geoutils.Raster` inheritance, which extends into other packages such as [xDEM](https://xdem.readthedocs.io/en/latest/index.html)
for analyzing digital elevation models. 

```{eval-rst}
.. inheritance-diagram:: geoutils.georaster.raster geoutils.georaster.satimg xdem.dem.DEM
    :top-classes: geoutils.georaster.raster.Raster
```

## The internal {class}`~geoutils.SatelliteImage` subclass

GeoUtils subclasses {class}`Rasters<geoutils.Raster>` to {class}`SatelliteImages<geoutils.SatelliteImage>` for remote sensing users interested in parsing 
metadata from space- or airborne imagery.

Based on the filename, or auxiliary files, the {class}`~geoutils.SatelliteImage` class attempts to automatically parse a `.datetime`, `.sensor`, `tile_name`,
and other information.


```{code-cell} ipython3
:tags: [hide-output]

import geoutils as gu

# Initiate a geo-image from disk
geoimg = gu.SatelliteImage(gu.examples.get_path("exploradores_aster_dem"))
geoimg
```
