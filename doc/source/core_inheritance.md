---
file_format: mystnb
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: geoutils-env
  language: python
  name: geoutils
---
(core-inheritance)=
# Inheritance to DEMs and beyond

Inheritance is practical to naturally pass down parent methods and attributes to child classes.

Many subtypes of {class}`Rasters<geoutils.Raster>` geospatial data exist that require additional attributes and methods, yet might benefit from methods
implemented in GeoUtils.

## Overview of {class}`~geoutils.Raster` inheritance

Current {class}`~geoutils.Raster` inheritance extends into other packages, such as [xDEM](https://xdem.readthedocs.io/)
for analyzing digital elevation models.

```{eval-rst}
.. inheritance-diagram:: geoutils.raster.raster
    :top-classes: geoutils.raster.raster.Raster
```

```{note}
The {class}`~xdem.DEM` class re-implements all methods of [gdalDEM](https://gdal.org/programs/gdaldem.html) (and more) to derive topographic attributes
(hillshade, slope, aspect, etc), coded directly in Python for scalability and tested to yield the exact same results.
Among others, it also adds a {attr}`~xdem.DEM.vcrs` property to consistently manage vertical referencing (ellipsoid, geoids).

If you are DEM-enthusiastic, **[check-out our sister package xDEM](https://xdem.readthedocs.io/) for digital elevation models.**
```

## And beyond

Many types of geospatial data can be viewed as a subclass of {class}`Rasters<geoutils.Raster>`, which have more attributes and require their own methods:
**spectral images**, **velocity fields**, **phase difference maps**, etc...

If you are interested to build your own subclass of {class}`~geoutils.Raster`, you can take example of the structure of {class}`xdem.DEM`.
Then, just add any of your own attributes and methods, and overload parent methods if necessary! Don't hesitate to reach out on our
GitHub if you have a subclassing project.
