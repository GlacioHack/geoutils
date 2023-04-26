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
(satimg-class)=

# The geo-image ({class}`~geoutils.SatelliteImage`)

Below, a summary of the {class}`~geoutils.SatelliteImage` object and its methods.

## Object definition

A {class}`~geoutils.SatelliteImage` is a subclass of {class}`~geoutils.Raster` that contains all its main attributes, and retains additional ones:
- a {class}`numpy.datetime64` as {class}`~geoutils.SatelliteImage.datetime`, and
- several {class}`strings<str>` for {class}`~geoutils.SatelliteImage.satellite`,  {class}`~geoutils.SatelliteImage.sensor`, {class}`~geoutils.SatelliteImage.version`, {class}`~geoutils.SatelliteImage.tile_name` and {class}`~geoutils.SatelliteImage.product`.

A {class}`~geoutils.SatelliteImage` also inherits the same derivative attributes as a {class}`~geoutils.Raster`.

## Metadata parsing

The {class}`~geoutils.SatelliteImage` attempts to parse metadata from the filename.

Right now are supported:
 - Landsat,
 - Sentinel-2,
 - SPOT,
 - ASTER,
 - ArcticDEM and REMA,
 - ALOS,
 - SRTM,
 - TanDEM-X, and
 - NASADEM.

The {class}`~geoutils.SatelliteImage.datetime` is always parsed or deduced.

For tiled products such as SRTM, the tile naming is also retrieved, which can be converted to geographic extent with {func}`geoutils.raster.satimg.parse_tile_attr_from_name`.

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

```{important}
The {class}`~geoutils.SatelliteImage` class is still in development, and we hope to further refine it in the future using metadata classes able to parse
auxiliary files metadata (such as [here](https://github.com/jlandmann/glaciersat/blob/master/glaciersat/core/imagery.py)).
```
