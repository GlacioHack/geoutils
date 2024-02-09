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
(proj-tools)=

# Projection tools

This section describes projection tools that are common to {class}`Rasters<geoutils.Raster>` and {class}`Vectors<geoutils.Vector>`, and facilitate
geospatial analysis.

## Get a metric coordinate system

A local metric coordinate system can be estimated for both {class}`Rasters<geoutils.Raster>` and {class}`Vectors<geoutils.Vector>` through the
{func}`~geoutils.Raster.get_metric_crs` function.

The metric system returned can be either "universal" (zone of the Universal Transverse Mercator or Universal Polar Stereographic system), or "custom"
(Mercator or Polar projection centered on the {class}`Raster<geoutils.Raster>` or {class}`Vector<geoutils.Vector>`).

```{code-cell} ipython3
import geoutils as gu

# Initiate a raster from disk
rast = gu.Raster(gu.examples.get_path("exploradores_aster_dem"))
rast.info()

# Estimate a universal metric CRS for the raster
rast.get_metric_crs()
```

## Get projected bounds

Projected bounds can be directly derived from both {class}`Rasters<geoutils.Raster>` and {class}`Vectors<geoutils.Vector>` through the
{func}`~geoutils.Raster.get_bounds_projected` function.

```{code-cell} ipython3
# Get raster bounds in geographic CRS by passing its EPSG code
rast.get_bounds_projected(4326)
```

```{important}
When projecting to a new CRS, the footprint shape of the data is generally deformed. To account for this, use {func}`~geoutils.Raster.get_footprint_projected` described below.
```

## Get projected footprint

A projected footprint can be derived from both {class}`Rasters<geoutils.Raster>` and {class}`Vectors<geoutils.Vector>` through the
{func}`~geoutils.Raster.get_footprint_projected` function.

For this, the original rectangular footprint polygon lines are densified to respect the deformation during reprojection.

```{code-cell} ipython3
# Get raster footprint in geographic CRS
rast_footprint = rast.get_footprint_projected(4326)

rast_footprint.plot()
```

This is for instance useful to check for intersection with other data.
