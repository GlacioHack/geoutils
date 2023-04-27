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
# Inheritance to geo-images and beyond

Inheritance is practical to naturally pass down parent methods and attributes to child classes.

Many subtypes of {class}`Rasters<geoutils.Raster>` geospatial data exist that require additional attributes and methods, yet might benefit from methods
implemented in GeoUtils.

## Overview of {class}`~geoutils.Raster` inheritance


Below is a diagram showing current {class}`~geoutils.Raster` inheritance, which extends into other packages such as [xDEM](https://xdem.readthedocs.io/)
for analyzing digital elevation models.

```{eval-rst}
.. inheritance-diagram:: geoutils.raster.raster geoutils.raster.satimg
    :top-classes: geoutils.raster.raster.Raster
```

```{note}
The {class}`~xdem.DEM` class re-implements all methods of [gdalDEM](https://gdal.org/programs/gdaldem.html) (and more) to derive topographic attributes
(hillshade, slope, aspect, etc), coded directly in Python for scalability and tested to yield the exact same results.
Among others, it also adds a {attr}`~xdem.DEM.vcrs` property to consistently manage vertical referencing (ellipsoid, geoids).

If you are DEM-enthusiastic, **[check-out our sister package xDEM](https://xdem.readthedocs.io/) for digital elevation models.**
```

## The internal {class}`~geoutils.SatelliteImage` subclass

GeoUtils subclasses {class}`Rasters<geoutils.Raster>` to {class}`SatelliteImages<geoutils.SatelliteImage>` for remote sensing users interested in parsing
metadata from space- or airborne imagery.

Based on the filename, or auxiliary files, the {class}`~geoutils.SatelliteImage` class attempts to automatically parse a
{attr}`~geoutils.SatelliteImage.datetime`, {attr}`~geoutils.SatelliteImage.sensor`, {attr}`~geoutils.SatelliteImage.tile_name`,
and other information.

```{code-cell} ipython3
import geoutils as gu

# Instantiate a geo-image from an ASTER image
filename_geoimg = gu.examples.get_path("exploradores_aster_dem")
geoimg = gu.SatelliteImage(filename_geoimg, silent=False)
```

```{code-cell} ipython3
# Instantiate a geo-image from a Landsat 7 image
filename_geoimg2 = gu.examples.get_path("everest_landsat_b4")
geoimg2 = gu.SatelliteImage(filename_geoimg2, silent=False)
```

Along these additional attributes, the {class}`~geoutils.SatelliteImage` possesses the same main attributes as a {class}`~geoutils.Raster`.

```{code-cell} ipython3

# The geo-image main attributes
geoimg
```

## And beyond

Many types of geospatial data can be viewed as a subclass of {class}`Rasters<geoutils.Raster>`, which have more attributes and require their own methods:
**spectral images**, **velocity fields**, **phase difference maps**, etc...

If you are interested to build your own subclass of {class}`~geoutils.Raster`, you can take example of the structure of {class}`geoutils.SatelliteImage` and
{class}`xdem.DEM`. Then, just add any of your own attributes and methods, and overload parent methods if necessary! Don't hesitate to reach out on our
GitHub if you have a subclassing project.
