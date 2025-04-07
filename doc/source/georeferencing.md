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

Below, a summary of the **georeferencing attributes** of geospatial data objects and the **methods to manipulate these
georeferencing attributes** in different projections, without any data transformation. For georeferenced transformations
(such as reprojection, cropping), see {ref}`geotransformations`.

```{code-cell} ipython3
:tags: [remove-cell]

# To get a good resolution for displayed figures
from matplotlib import pyplot
pyplot.rcParams['figure.dpi'] = 600
pyplot.rcParams['savefig.dpi'] = 600
pyplot.rcParams['font.size'] = 9
```

## Attributes

In GeoUtils, the **georeferencing syntax is consistent across all geospatial data objects**. Additionally, **data objects
load only their metadata by default**, allowing quick operations on georeferencing without requiring the array data
(for a {class}`~geoutils.Raster`) to be present in memory.

### Metadata summary

To summarize all the metadata of a geospatial data object, including its georeferencing, {func}`~geoutils.Raster.info` can be used:


```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the code for opening example files"
:  code_prompt_hide: "Hide the code for opening example files"

import geoutils as gu
rast = gu.Raster(gu.examples.get_path("exploradores_aster_dem"))
vect = gu.Vector(gu.examples.get_path("exploradores_rgi_outlines"))
```

```{code-cell} ipython3
# Print raster info
rast.info()
```

```{code-cell} ipython3
# Print vector info
vect.info()
```

### Coordinate reference systems

[Coordinate reference systems (CRSs)](https://en.wikipedia.org/wiki/Spatial_reference_system), sometimes also called
spatial reference systems (SRSs), define the 2D projection of the geospatial data. They are stored as a
{class}`pyproj.crs.CRS` object in {attr}`~geoutils.Raster.crs`.

```{code-cell} ipython3
# Show CRS attribute of raster
print(rast.crs)
```
```{code-cell} ipython3
# Show CRS attribute of vector as a WKT
print(vect.crs.to_wkt())
```

More information on the manipulation of {class}`pyproj.crs.CRS` objects can be found in [PyProj's documentation](https://pyproj4.github.io/pyproj/stable/).

```{note}
3D CRSs for elevation data are only emerging, and not consistently defined in the metadata.
The [vertical referencing functionalities of xDEM](https://xdem.readthedocs.io/en/stable/vertical_ref.html)
can help define a 3D CRS.
```

(bounds)=
### Bounds

Bounds define the spatial extent of geospatial data, composed of the "left", "right", "bottom" and "top" coordinates.
The {attr}`~geoutils.Raster.bounds` of a raster or a vector is a {class}`rasterio.coords.BoundingBox` object:

```{code-cell} ipython3
# Show bounds attribute of raster
rast.bounds
```
```{code-cell} ipython3
# Show bounds attribute of vector
vect.bounds
```

```{note}
To define {attr}`~geoutils.Raster.bounds` consistently between rasters and vectors, {attr}`~geoutils.Vector.bounds`
 corresponds to {attr}`geopandas.GeoSeries.total_bounds` (total bounds of all geometry features) converted to a {class}`rasterio.coords.BoundingBox`.

To reproduce the behaviour of {attr}`geopandas.GeoSeries.bounds` (per-feature bounds) with a
{class}`~geoutils.Vector`, use {attr}`~geoutils.Vector.geom_bounds`.
```

### Footprints

As reprojections between CRSs deform shapes, including extents, it is often better to consider a vectorized footprint
to calculate intersections in different projections. The {class}`~geoutils.Raster.footprint` is a
{class}`~geoutils.Vector` object with a single polygon geometry for which points have been densified, allowing
reliable computation of extents between CRSs.

```{code-cell} ipython3
# Print raster footprint
rast.get_footprint_projected(rast.crs)
```
```{code-cell} ipython3
# Plot vector footprint
vect.get_footprint_projected(vect.crs).plot()
```

### Grid (only for rasters)

A raster's grid origin and resolution are defined by its geotransform attribute, {attr}`~geoutils.Raster.transform`.
Combined with the 2D shape of the data array {attr}`~geoutils.Raster.shape` (and independently of the number of
bands {attr}`~geoutils.Raster.bands`), these two attributes define the georeferenced grid of a raster.

From it are derived the resolution {attr}`~geoutils.Raster.res`, and {attr}`~geoutils.Raster.height` and
{attr}`~geoutils.Raster.width`, as well as the bounds detailed above in {ref}`bounds`.

```{code-cell} ipython3
# Get raster transform and shape
print(rast.transform)
print(rast.shape)
```

(pixel-interpretation)=
### Pixel interpretation (only for rasters)

A largely overlooked aspect of a raster's georeferencing is the pixel interpretation stored in the
[AREA_OR_POINT metadata](https://gdal.org/user/raster_data_model.html#metadata).
Pixels can be interpreted either as **"Area"** (the most common) where **the value represents a sampling over the region
of the pixel (and typically refers to the upper-left corner coordinate)**, or as **"Point"**
where **the value relates to a point sample (and typically refers to the center of the pixel)**, the latter often used
for digital elevation models (DEMs).

Pixel interpretation is stored as a string in the {attr}`geoutils.Raster.area_or_point` attribute.

```{code-cell} ipython3
# Get pixel interpretation of raster
rast.area_or_point
```

Although this interpretation is not intended to influence georeferencing, it **can influence sub-pixel coordinate
interpretation during analysis**, especially for raster–vector–point interfacing operations such as point interpolation,
or re-gridding, and might also be a problem if defined differently when comparing two rasters.

```{important}
By default, **pixel interpretation induces a half-pixel shift during raster–point interfacing for a "Point" interpretation**
(mirroring [GDAL's default ground-control point behaviour](https://trac.osgeo.org/gdal/wiki/rfc33_gtiff_pixelispoint)),
but only **raises a warning for raster–raster operations** if interpretations differ.

This behaviour can be modified at the package-level by using GeoUtils' {ref}`config`
`shift_area_or_point` and `warns_area_or_point`.
```

## Manipulation

Several functionalities are available to facilitate the manipulation of the georeferencing.

### Getting projected bounds and footprints

Retrieving projected bounds or footprints in any CRS is possible using directly {func}`~geoutils.Raster.get_bounds_projected`
and {func}`~geoutils.Raster.get_footprint_projected`.

```{code-cell} ipython3
# Get footprint of larger buffered vector in polar stereo CRS (to show deformations)
vect.buffer_metric(10**6).get_footprint_projected(3995).plot()
```

### Getting a metric CRS

A local metric coordinate system can be estimated for both {class}`Rasters<geoutils.Raster>` and {class}`Vectors<geoutils.Vector>` through the
{func}`~geoutils.Raster.get_metric_crs` function.

The metric system returned can be either "universal" (zone of the Universal Transverse Mercator or Universal Polar Stereographic system), or "custom"
(Mercator or Polar projection centered on the {class}`Raster<geoutils.Raster>` or {class}`Vector<geoutils.Vector>`).

```{code-cell} ipython3
# Get local metric CRS
rast.get_metric_crs()
```

### Re-set georeferencing metadata

The georeferencing metadata of an object can be re-set (overwritten) by setting the corresponding attribute such as {func}`geoutils.Vector.crs` or
{func}`geoutils.Raster.transform`. When specific options might be useful during setting, a set function exists,
such as for {func}`geoutils.Raster.set_area_or_point`.

```{warning}
Re-setting should only be used if the **data was erroneously defined and needs to be corrected in-place**.
To create geospatial data from its attributes, use the construction functions such as {func}`~geoutils.Raster.from_array`.
```

```{code-cell} ipython3
# Re-set CRS
import pyproj
rast.crs = pyproj.CRS(4326)
rast.crs
```

```{code-cell} ipython3
# Re-set pixel interpretation
rast.set_area_or_point("Point")
rast.area_or_point
```


### Coordinates to indexes (only for rasters)

Raster grids are notoriously unintuitive to manipulate on their own due to the Y axis being inverted and stored as first axis.
GeoUtils' features account for this under-the-hood when plotting, interpolating, gridding, or performing any other operation involving the raster coordinates.

Three functions facilitate the manipulation of coordinates, while respecting any {ref}`Pixel interpretation<pixel-interpretation>`:

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
