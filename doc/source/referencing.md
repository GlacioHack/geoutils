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
(referencing)=
# Referencing

Below, a summary of the **referencing attributes** of geospatial data objects and the **methods to manipulate these**
in different projections, without any data transformation.

For georeferenced transformations (such as reprojection, or polygonization), see {ref}`transformations`.

```{code-cell} ipython3
:tags: [remove-cell]

# To get a good resolution for displayed figures
from matplotlib import pyplot
pyplot.rcParams['figure.dpi'] = 600
pyplot.rcParams['savefig.dpi'] = 600
pyplot.rcParams['font.size'] = 9
```

## Attributes

In GeoUtils, the **georeferencing syntax is consistent** across all geospatial data objects. Additionally, data objects
**load only their metadata by default**, allowing quick operations on georeferencing without requiring the array data
(for a {class}`~geoutils.Raster`) to be present in memory.

### Metadata summary

{meth}`ds.rst.info() or Raster.info() <RasterBase.info>`<br>
{meth}`gdf.vct.info() or Vector.info() <VectorBase.info>`<br>
{meth}`gdf.pc.info() or PointCloud.info() <geoutils.pointcloud.base.GeometryPointCloudBase.info>`

The {meth}`~RasterBase.info` method summarizes **all metadata** of a geospatial data object, including its georeferencing.

```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the code for opening example files"
:  code_prompt_hide: "Hide the code for opening example files"

import geoutils as gu
ds = gu.open_raster(gu.examples.get_path("exploradores_aster_dem"))
gdf = gu.open_vector(gu.examples.get_path("exploradores_rgi_outlines"))
```

```{code-cell} ipython3
# Print raster info
ds.rst.info()
```

```{code-cell} ipython3
# Print vector info
gdf.vct.info()
```

### Coordinate reference systems

{attr}`ds.rst.crs or Raster.crs <RasterBase.crs>`<br>
{attr}`gdf.vct.crs or Vector.crs <VectorBase.crs>`<br>
{attr}`gdf.pc.crs or PointCloud.crs <PointCloudBase.crs>`

[Coordinate reference systems (CRSs)](https://en.wikipedia.org/wiki/Spatial_reference_system), sometimes also called
spatial reference systems (SRSs), **define the 2D or 3D projection of geospatial data**. GeoUtils stores them as
{class}`pyproj.crs.CRS` objects.

```{code-cell} ipython3
# Show CRS attribute of raster
print(ds.rst.crs)
```
```{code-cell} ipython3
# Show CRS attribute of vector as a WKT
print(gdf.vct.crs.to_wkt())
```

More information on the manipulation of {class}`pyproj.crs.CRS` objects can be found in [PyProj's documentation](https://pyproj4.github.io/pyproj/stable/).

```{note}
3D CRSs for elevation data are only emerging, and not consistently defined in the metadata.
The [vertical referencing functionalities of xDEM](https://xdem.readthedocs.io/en/stable/vertical_ref.html)
can help define a 3D CRS.
```

(bounds)=
### Bounding boxes

{attr}`ds.rst.bbox or Raster.bbox <RasterBase.bbox>`<br>
{attr}`gdf.vct.bbox or Vector.bbox <VectorBase.bbox>`<br>
{attr}`gdf.pc.bbox or PointCloud.bbox <geoutils.pointcloud.base.GeometryPointCloudBase.bbox>`

Bounding boxes define the **spatial extent** of geospatial data through the "left", "right", "bottom", and "top"
coordinates. GeoUtils represents them with {class}`rasterio.coords.BoundingBox` objects.

```{code-cell} ipython3
# Show bounding box of raster
ds.rst.bbox
```
```{code-cell} ipython3
# Show bounding box of vector
gdf.vct.bbox
```

```{note}
To define {attr}`~RasterBase.bbox` consistently between rasters and vectors, {attr}`~VectorBase.bbox`
 corresponds to {attr}`geopandas.GeoSeries.total_bounds` (total bounds of all geometry features) converted to a {class}`rasterio.coords.BoundingBox`.

To reproduce the behaviour of {attr}`geopandas.GeoSeries.bounds` (per-feature bounds) with a
{class}`~geoutils.Vector`, use {attr}`~geoutils.Vector.geom_bounds`.
```

### Footprints

{attr}`ds.rst.footprint or Raster.footprint <RasterBase.footprint>`<br>
{attr}`gdf.vct.footprint or Vector.footprint <VectorBase.footprint>`<br>
{attr}`gdf.pc.footprint or PointCloud.footprint <geoutils.pointcloud.base.GeometryPointCloudBase.footprint>`

A footprint defines **a polygon of the object extent**, vectorized with densified lines.
As reprojections between CRSs deform shapes, a footprint provides more reliable intersections than a bounding box.

```{code-cell} ipython3
# Plot the raster and vector footprints together
_, ax = pyplot.subplots()
ds.rst.footprint.vct.plot(ax=ax, fc="none", ec="tab:blue", lw=2)
gdf.vct.footprint.vct.reproject(ds).vct.plot(ax=ax, fc="none", ec="tab:orange", lw=2)
_ = ax.set_title("Raster (blue) and vector (orange) footprints")
```

### Grid (only for rasters)

{attr}`ds.rst.transform or Raster.transform <RasterBase.transform>`<br>
{attr}`ds.rst.shape or Raster.shape <RasterBase.shape>`

The {attr}`~RasterBase.transform` and {attr}`~RasterBase.shape` attributes define a **raster's georeferenced grid through its origin, resolution and 2D array shape**, independently
of the number of bands {attr}`~RasterBase.bands`.

From the grid are derived the resolution {attr}`~RasterBase.res`, and {attr}`~RasterBase.height` and
{attr}`~RasterBase.width`, as well as the bounding box detailed above in {ref}`bounds`.

```{code-cell} ipython3
# Get raster transform and shape
print(ds.rst.transform)
print(ds.rst.shape)
```

(pixel-interpretation)=
### Pixel interpretation (only for rasters)

{attr}`ds.rst.area_or_point or Raster.area_or_point <RasterBase.area_or_point>`

A largely overlooked aspect of a raster's georeferencing is the **pixel interpretation** stored in the
[AREA_OR_POINT metadata](https://gdal.org/user/raster_data_model.html#metadata).
Pixels can be interpreted either as **"Area"** (the most common) where **the value represents a sampling over the region
of the pixel** (and typically refers to the upper-left corner coordinate), or as **"Point"**
where **the value relates to a point sample** (and typically refers to the center of the pixel), the latter often used
for digital elevation models (DEMs).

Pixel interpretation is stored as a string.

```{code-cell} ipython3
# Get pixel interpretation of raster
ds.rst.area_or_point
```

Although this interpretation is not intended to influence georeferencing, it **can influence sub-pixel coordinate
interpretation during analysis**, especially for raster–vector–point interfacing operations such as point interpolation,
or re-gridding, and might also be a problem if defined differently when comparing two rasters.

```{important}
By default, **pixel interpretation induces a half-pixel shift during raster–point interfacing for a "Point" interpretation**
(mirroring [GDAL's default ground-control point behaviour](https://trac.osgeo.org/gdal/wiki/rfc33_gtiff_pixelispoint)),
but only **raises a warning for raster–raster operations** if interpretations differ.

This behaviour can be modified at the package-level by using GeoUtils' {ref}`config`
`shift_area_or_point` and `warn_area_or_point`.
```

## Manipulation

Several functionalities are available to facilitate the manipulation of the georeferencing.

### Projected bounding boxes

{meth}`ds.rst.get_bbox_projected() or Raster.get_bbox_projected() <RasterBase.get_bbox_projected>`<br>
{meth}`gdf.vct.get_bbox_projected() or Vector.get_bbox_projected() <VectorBase.get_bbox_projected>`<br>
{meth}`gdf.pc.get_bbox_projected() or PointCloud.get_bbox_projected() <VectorBase.get_bbox_projected>`

These methods return the **bounding box in another CRS**, accounting for non-linear deformation by densifying its edges
during projection.

### Projected footprints

{meth}`ds.rst.get_footprint_projected() or Raster.get_footprint_projected() <RasterBase.get_footprint_projected>`<br>
{meth}`gdf.vct.get_footprint_projected() or Vector.get_footprint_projected() <VectorBase.get_footprint_projected>`<br>
{meth}`gdf.pc.get_footprint_projected() or PointCloud.get_footprint_projected() <VectorBase.get_footprint_projected>`

These methods return **the footprint polygon in another CRS**, also densifying lines to preserve shape during projection.

```{code-cell} ipython3
# Get footprint of larger buffered vector in polar stereo CRS (to show deformations)
gdf.vct.buffer_metric(10**6).vct.get_footprint_projected(3995).vct.plot()
```

### Metric CRS

{meth}`ds.rst.get_metric_crs() or Raster.get_metric_crs() <RasterBase.get_metric_crs>`<br>
{meth}`gdf.vct.get_metric_crs() or Vector.get_metric_crs() <VectorBase.get_metric_crs>`<br>
{meth}`gdf.pc.get_metric_crs() or PointCloud.get_metric_crs() <VectorBase.get_metric_crs>`

These methods estimate a **local metric coordinate system**.

The local coordinate system can be **either "universal"**, a standard zone of the Universal
Transverse Mercator (UTM) or Universal Polar Stereographic (UPS) system, **or "custom"**
(a stereographic projection centered exactly on the centroid of the {class}`Raster<geoutils.Raster>`,
{class}`Vector<geoutils.Vector>` or {class}`PointCloud<geoutils.PointCloud>`).

The custom CRS has true scale near its center, suiting small regions particularly well.

```{code-cell} ipython3
# Get local metric CRS
ds.rst.get_metric_crs()
```

### Edit raster metadata

{meth}`ds.rst.edit() or Raster.edit() <RasterBase.edit>`

This method returns **a copy with edited metadata fields**, while keeping the source and its pixel
values unchanged. Omitted fields keep their current values, explicit `None` values clear optional metadata, and new
tags are merged with the existing tags.

```{warning}
Editing or resetting georeferencing metadata should only be used if the **data was erroneously defined and needs to be
corrected**. To create geospatial data from its attributes, use construction methods such as
{meth}`from_array() <RasterBase.from_array>`.
```

```{code-cell} ipython3
# Correct several metadata fields on a new raster
edited_ds = ds.rst.edit(tags={"purpose": "corrected metadata"}, area_or_point="Point")
(edited_ds.rst.tags, edited_ds.rst.area_or_point)
```

### Set individual raster metadata

{meth}`ds.rst.set_crs() or Raster.set_crs() <RasterBase.set_crs>`<br>
{meth}`ds.rst.set_transform() or Raster.set_transform() <RasterBase.set_transform>`<br>
{meth}`ds.rst.set_nodata() or Raster.set_nodata() <RasterBase.set_nodata>`<br>
{meth}`ds.rst.set_area_or_point() or Raster.set_area_or_point() <RasterBase.set_area_or_point>`

These methods **update one metadata field in-place** and provide field-specific options when needed. Assigning the corresponding
attribute, such as {attr}`~RasterBase.crs` or {attr}`~RasterBase.transform`, uses the setter's default options.

```{code-cell} ipython3
# Correct one metadata field in place
edited_ds.rst.set_area_or_point("Area")
edited_ds.rst.area_or_point
```


### Coordinates to indexes (only for rasters)

{meth}`ds.rst.xy2ij() or Raster.xy2ij() <RasterBase.xy2ij>`<br>
{meth}`ds.rst.ij2xy() or Raster.ij2xy() <RasterBase.ij2xy>`<br>
{meth}`ds.rst.coords() or Raster.coords() <RasterBase.coords>`

Raster grids are notoriously unintuitive to manipulate on their own due to the Y axis being inverted and stored as first axis.
GeoUtils' features account for this under-the-hood when plotting, interpolating, gridding, or performing any other operation involving the raster coordinates.

These methods convert between coordinates and array indices, or return coordinates in the order of the data array
axes, possibly as a meshgrid. They respect any {ref}`Pixel interpretation<pixel-interpretation>`.

```{code-cell} ipython3
# Get coordinates from row/columns indices
x, y = ds.rst.ij2xy(i=[0, 1], j=[2, 3])
(x, y)
```

```{code-cell} ipython3
# Get indices from coordinates
i, j = ds.rst.xy2ij(x=x, y=y)
(i, j)
```

```{code-cell} ipython3
:tags: [hide-output]
# Get vector X/Y coordinates corresponding to data array
ds.rst.coords(grid=False)
```
