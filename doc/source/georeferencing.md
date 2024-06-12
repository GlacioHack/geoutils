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
(georeferencing)=
# Referencing

Below, a summary of the **georeferencing attribute definition** of geospatial data objects and the **methods to manipulate 
georeferencing attributes** in different projections, without any data transformation. For georeferenced transformation 
(reprojection, cropping), see {ref}`geotransformations`.

# Attributes

In GeoUtils, the georeferencing syntax is consistent across all geospatial data objects. Additionally, data objects 
load only their metadata by default, allowing quick operations on georeferencing without requiring the array data 
(for a {class}`~geoutils.Raster`) or geometry data (for a {class}`~geoutils.Vector`) to be present in memory.

```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the code for opening example files"
:  code_prompt_hide: "Hide the code for opening example files"

import geoutils as gu
rast = gu.Raster(gu.examples.get_path("exploradores_aster_dem"))
vect = gu.Vector(gu.examples.get_path("exploradores_rgi_outlines"))
```

## Metadata summary

To summarize all the metadata of a geospatial data object, including its georeferencing, {func}`~geoutils.Raster.info` can be used:

```{code-cell} ipython3
# Print raster and vector info
rast.info()
vect.info()
```

## Coordinate reference systems

[Coordinate reference systems (CRSs)](https://en.wikipedia.org/wiki/Spatial_reference_system), sometimes also called a 
spatial reference systems (SRSs), define the 2D projection of the geospatial data. They are stored as a 
{class}`pyproj.crs.CRS` object in {attr}`~geoutils.Raster.crs`. More information on the manipulation 
of {class}`pyproj.crs.CRS` objects can be found in [PyProj's documentation](https://pyproj4.github.io/pyproj/stable/).

```{code-cell} ipython3
# Show CRS attribute of a raster and vector
print(rast.crs)
print(vect.crs)
```

```{note}
3D CRSs for elevation data are only emerging, and not consistently defined in the metadata.
The [vertical referencing functionalities of xDEM](https://xdem.readthedocs.io/en/stable/vertical_ref.html) 
can help define a 3D CRS.
```

## Bounds

Bounds define the spatial extent of geospatial data, and are often useful to calculate extent intersections, or to reproject or crop on a matching extent.
The {attr}`~geoutils.Raster.bounds` of a raster of vector correspond to a {class}`rasterio.coords.BoundingBox` object, composed of the "left", "right", "bottom" and "top" coordinates making up the bounds.

```{code-cell} ipython3
# Show bounds attribute of a raster and vector
print(rast.bounds)
print(vect.bounds)
```

```{note}
To define {attr}`~geoutils.Raster.bounds` consistently between rasters and vectors, the {attr}`~geoutils.Vector.bounds` 
attribute corresponds to the {attr}`geopandas.GeoDataFrame.total_bounds` attribute (bounds of all geometries).

To derive per-geometry bounds with a {class}`~geoutils.Vector` (matching {attr}`geopandas.GeoDataFrame.bounds`), use 
{attr}`~geoutils.Vector.geom_bounds`.
```

## Footprints

Reprojections between CRSs deform shapes, including raster or vector extents, thus it is often better to consider a 
vectorized footprint. The {class}`~geoutils.Raster.footprint` is a {class}`~geoutils.Vector` object with a single polygon 
geometry for which points have been densified, allowing reliable computation of extents between CRSs.

```{code-cell} ipython3
# Show bounds attribute of a raster and vector
print(rast.get_footprint_projected(rast.crs))
print(vect.get_footprint_projected(vect.crs))
```

## Grid (only for rasters)

A raster's grid origin and resolution are defined by its geotransform attribute, {attr}`~geoutils.Raster.transform`, and its shape by the data array shape.
From it are derived the resolution {attr}`~geoutils.Raster.res`, the 2D raster shape 
{attr}`~geoutils.Raster.shape` made up of its {attr}`~geoutils.Raster.height` and {attr}`~geoutils.Raster.width`.

```{code-cell} ipython3
# Get raster transform and shape
print(rast.transform)
print(rast.shape)
```

(pixel-interpretation)=
## Pixel interpretation (only for rasters)

A largely overlooked aspect of a raster's georeferencing is the pixel interpretation stored in the 
[AREA_OR_POINT metadata](https://gdal.org/user/raster_data_model.html#metadata). 
Pixels can be interpreted either as **"Area"** (the most common) where **the value represents a sampling over the region 
of the pixel (and typically refers to the upper-left corner coordinate)**, or as **"Point"**
where **the value relates to a point sample (and typically refers to the center of the pixel)**, used often for DEMs. 
Although this interpretation is not intended to influence georeferencing, it **can influence sub-pixel coordinate
interpretation during analysis**, especially for raster–vector–point interfacing operations such as point interpolation, 
or re-gridding, and might also be a problem if defined differently when comparing two rasters.

Pixel interpretation is stored as a string in the {attr}`geoutils.Raster.area_or_point` attribute. 

```{code-cell} ipython3
# Get pixel interpretation of raster
rast.area_or_point
```

```{important}
By default, **pixel interpretation can induce a half-pixel shift during raster–point interfacing** 
(mirroring [GDAL's default ground-control point behaviour](https://trac.osgeo.org/gdal/wiki/rfc33_gtiff_pixelispoint)), 
but only **raises a warning for raster–raster operations**.

This behaviour can be modified at the package-level by using [GeoUtils' configuration parameters]() 
`shift_area_or_point` and `warns_area_or_point`.
```

# Manipulation

Several functionalities are available to facilitate the manipulation of the georeferencing.

## From coordinates to raster indexes, and vice-versa

Raster grids are notoriously unintuitive to manipulate on their own due to the Y axis being inverted and stored as first axis.
GeoUtils' features account for this under-the-hood when plotting, interpolating, gridding, or performing any other operation involving the raster coordinates.

Three functions facilitate the manipulation of coordinates, while respecting any {ref}`pixel-interpretation`:

1. {func}`~geoutils.Raster.xy2ij` to convert array indices to coordinates,
2. {func}`~geoutils.Raster.ij2xy` to convert coordinates to array indices (reversible with {func}`~geoutils.Raster.xy2ij` for any pixel interpretation),
3. {func}`~geoutils.Raster.coords` to directly obtain coordinates in order corresponding to the data array axes, possibly as a meshgrid.

```{code-cell} ipython3
# Get coordinates from row/columns indices
x, y = rast.ij2xy(i=[0, 1], j=[2, 3])
(x, y)
```

```{code-cell} ipython3
# Get indices from coordinates
i, j = rast.xy2ij(x=x, y=y)
(i, j)
```

```{code-cell} ipython3
:tags: [hide-output]
# Get vector X/Y coordinates corresponding to data array
rast.coords(grid=False)
```

## Getting projected bounds and footprints

Retrieving projected bounds or footprints in any CRS is possible using directly {func}`~geoutils.Raster.get_bounds_projected` 
and {func}`~geoutils.Raster.get_footprint_projected`.

```{code-cell} ipython3
# Get raster footprint in geographic CRS
rast_footprint = rast.get_footprint_projected(4326)
rast_footprint.plot()
```

## Getting a metric CRS

A metric CRS at the location of the geospatial data can be derived using {func}`geoutils.Raster.get_metric_crs`.

```{code-cell} ipython3
# Get local metric CRS
print(vect.get_metric_crs())
print(rast.get_metric_crs())
```

## Re-set georeferencing metadata

{func}`geoutils.Vector.set_crs`
{func}`geoutils.Raster.set_transform`
{func}`geoutils.Raster.set_area_or_point`



