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
(satimg-parsing)=

# Sensor metadata parsing

GeoUtils functionalities for remote sensing users interested in parsing metadata from space- or airborne imagery.

## Parsing metadata at raster instantiation

A {class}`~geoutils.Raster` can be instantiated while trying to parse metadata usint the ``parse_sensor_metadata`` argument.

```{code-cell} ipython3
import geoutils as gu

# Parse metadata from an ASTER raster
filename_aster = gu.examples.get_path("exploradores_aster_dem")
rast_aster = gu.Raster(filename_aster, parse_sensor_metadata=True, silent=False)
```


```{code-cell} ipython3
# Parse metadata from a Landsat 7 raster
filename_landsat = gu.examples.get_path("everest_landsat_b4")
rast_landsat = gu.Raster(filename_landsat, parse_sensor_metadata=True, silent=False)
```

The metadata is then stored in the {attr}`~geoutils.Raster.tags` attribute of the raster.

```{code-cell} ipython3
rast_aster.tags
```

For tiled products such as SRTM, the tile naming is also retrieved, and converted to usable tile sizes and extents based on known metadata.

## Supported sensors

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

```{important}
Sensor metadata parsing is still in development. We hope to add the ability to parse from
auxiliary files in the future (such as [here](https://github.com/jlandmann/glaciersat/blob/master/glaciersat/core/imagery.py)).
```
