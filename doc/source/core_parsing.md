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

GeoUtils contains **basic functionalities to parse sensor metadata** from space- or airborne imagery.

Those rely only on filenames, parsing basic metadata such as date of acquisition, tile name, or product name.

```{seealso}
For more exhaustive support working with remote sensing data, we refer to packages such as [EOReader](https://eoreader.readthedocs.io/stable/) or [SatPy](https://github.com/pytroll/satpy).
```

## Parsing metadata at raster instantiation

A {class}`~geoutils.Raster` can parse metadata when instantiated with the ``parse_sensor_metadata`` argument.

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

The metadata is then stored in the {attr}`~RasterBase.tags` attribute of the raster.

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
