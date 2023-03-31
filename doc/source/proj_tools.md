---
file_format: mystnb
kernelspec:
  name: geoutils
---
(proj-tools)=

# Projection tools

This section describes projection tools that are common to {class}`Rasters<geoutils.Raster>` and {class}`Vectors<geoutils.Vector>`, and facilitate 
geospatial analysis.

## Get projected bounds

Projected bounds can be directly derived from both {class}`Rasters<geoutils.Raster>` and {class}`Vectors<geoutils.Vector>` through the
{func}`~geoutils.Raster.get_bounds_projected` function.

```{code-cell} ipython3
import geoutils as gu

# Initiate a raster from disk
rast = gu.Raster(gu.examples.get_path("exploradores_aster_dem"))
print(rast.info())

# Get raster bounds in geographic CRS by passing its EPSG code
rast.get_bounds_projected(4326)
```

```{important}
When projecting to a new CRS, the footprint shape of the data is generally deformed. To account for this, use {func}`~geoutils.Raster.
get_footprint_projected` described below.
```

## Get projected footprint

A projected footprint can be derived from both {class}`Rasters<geoutils.Raster>` and {class}`Vectors<geoutils.Vector>` through the
{func}`~geoutils.Raster.get_footprint_projected` function.

For this, the original rectangular footprint polygon lines are densified to respect the deformation during reprojection. 

```{code-cell} ipython3
# Get raster footprint in geographic CRS
rast_footprint = rast.get_footprint_projected(3412)

rast_footprint.show()
```

This is for instance useful to check for intersection with other data.

```{code-cell} ipython3
# Open a vector
vect = gu.Vector(gu.examples.get_path("exploradores_rgi_outlines"))

# Do these raster and vector intersect?
any(vect.intersects(rast_footprint))
```

## Find a local metric projection

## Merge or align multiple bounds

## Create a tiling

## Shape-preserving vector reprojection
