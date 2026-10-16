(feature-overview)=
# Feature and scalability overview

GeoUtils provides a unified API for manipulating **raster**, **vector**, and **point cloud** data, and provides **scalable CPU execution** for most raster and point cloud operations, as well as some vector operations, through Dask or multiprocessing.

As many of our numerical operations rely on NumPy, SciPy or Numba, those are planned to be linked to their **GPU counterparts** (CuPy and Numba CUDA) in the future.

The **{ref}`summary tables<tables-overview>` directly below** list the core features of GeoUtils and their scalability.

```{seealso}
If you are interested in **porting from GDAL/OGR**, see our {ref}`cheatsheet-osgeo` page.
While tables below provide a scalability summary, the detailed **scalable execution behaviour of all operations** is available on the {ref}`scalability-support` page.
For **performance comparisons**, see the {ref}`benchmarking-performance` page.
```

## Summary

GeoUtils exposes a **consistent API across raster, vector and point cloud objects** where possible (similar in spirit to the recent [GDAL CLI overhaul](https://gdal.org/en/stable/programs/index.html)). Many operations also support convenient **match-reference arguments** (e.g., matching a grid for reprojection or rasterization, bounds for cropping, or point coordinates for interpolation). See the {ref}`core-match-ref` page for details.

At its core, GeoUtils provides two interchangeable ways to work with geospatial data:

- **Accessors** that extend existing Xarray or GeoPandas structures ({class}`rst <geoutils.DataArrayRasterAccessor>` for rasters, {class}`pc <geoutils.GeoDataFramePointCloudAccessor>` for point clouds, and {class}`vct <geoutils.GeoDataFrameVectorAccessor>` for vectors),
- **GeoUtils objects** {class}`~geoutils.Raster`, {class}`~geoutils.PointCloud`, {class}`~geoutils.Vector`.

All raster and point cloud operations support **scalable execution** using [Dask](https://www.dask.org/) or multiprocessing, allowing large datasets to be processed **chunk-by-chunk without loading the full array into memory**. Some vector operations also support scalable execution, although vector inputs are often less limiting.

Additionally, some numerical routines of GeoUtils provide **multiple computational backends** (e.g., SciPy or Numba implementations).

All methods are tested to ensure they produce **identical results** whether executed in-memory, chunked, or with different computational backends.

(tables-overview)=
## Data operations

We first describe GeoUtils' core **data operations**, which operate on underlying arrays or geometries and can therefore **benefit from scalable execution**.

```{admonition} Legend
- **`/`** denotes methods **shared across object types**.
- **`⟷`** denotes methods **interfacing between two object types**.
```

```{list-table} Common API for data operations
:widths: 3 5 1
:header-rows: 1
:align: left
:class: tight-table

* - Method
  - Description
  - Scalable

* - <span class="gu-table-section">Raster / Vector / Point</span>
  -
  -

* - {meth}`plot() <RasterBase.plot>`
  - Display geospatial data.
  - ✅

* - {meth}`reproject() <RasterBase.reproject>`
  - Change CRS or raster grid.
  - ✅

* - {meth}`crop() <RasterBase.crop>`
  - Select data within a bounding box.
  - ✅

* - {meth}`clip() <RasterBase.clip>`
  - Mask, select or cut data by geometry.
  - ✅

* - {meth}`translate() <RasterBase.translate>`
  - Shift spatial coordinates.
  - ✅

* - <span class="gu-table-section">Raster / Point</span>
  -
  -

* - {meth}`create_mask() <VectorBase.create_mask>`
  - Create boolean masks from geometries.
  - ✅

* - {meth}`proximity() <RasterBase.proximity>`
  - Measure distance to target values.
  - ✅

* - {meth}`filter() <RasterBase.filter>`
  - Filter values over local neighborhoods.
  - ✅

* - {meth}`fill_nodata() <RasterBase.fill_nodata>` / {meth}`sieve() <RasterBase.sieve>`
  - Fill raster gaps or remove small raster regions.
  - ❌

* - {meth}`subsample() <RasterBase.subsample>`
  - Randomly select valid cells or points.
  - ✅

* - {meth}`cosample() <RasterBase.cosample>`
  - Select common valid locations across datasets.
  - ✅

* - {meth}`pairsample() <RasterBase.pairsample>`
  - Sample valid pairs across spatial distances.
  - ✅

* - {meth}`stats() <RasterBase.stats>`
  - Summarize valid values.
  - ✅

* - {meth}`stats() <RasterBase.stats>` with ``by``
  - Group statistics by bins, categories or zones.
  - ✅

* - {meth}`variogram() <RasterBase.variogram>`
  - Estimate and fit spatial semivariance.
  - ✅

* - {meth}`estimate_error_structure() <RasterBase.estimate_error_structure>`
  - Estimate error magnitude and spatial correlation.
  - ✅

* - {meth}`~RasterBase.random_field`
  - Generate spatially correlated random values.
  - ✅

* - <span class="gu-table-section">Raster ⟷ Vector</span>
  -
  -

* - {meth}`rasterize() <VectorBase.rasterize>`
  - Burn geometries onto a raster grid.
  - ✅

* - {meth}`rasterize() <VectorBase.rasterize>` with ``fractional=True``
  - Compute coverage fractions.
  - ✅

* - {meth}`stats() <RasterBase.stats>` with ``fractional=True``
  - Compute area-weighted polygon statistics.
  - ❌

* - {meth}`polygonize() <RasterBase.polygonize>`
  - Convert raster regions to polygons.*
  - ✅

* - <span class="gu-table-section">Raster ⟷ Point</span>
  -
  -

* - {meth}`interp_at_points() <RasterBase.interp_at_points>`
  - Interpolate raster values at points.
  - ✅

* - {meth}`reduce_at_points() <RasterBase.reduce_at_points>`
  - Aggregate raster values around points.
  - ✅

* - {meth}`to_pointcloud() <RasterBase.to_pointcloud>`
  - Convert raster cells to points.
  - ✅

* - {meth}`from_pointcloud_regular() <RasterBase.from_pointcloud_regular>`
  - Convert regular points to a raster.
  - ❌

* - {meth}`grid() <PointCloudBase.grid>`
  - Create a raster from irregular points.
  - ✅
```

## Metadata properties and operations

In addition to data operations, GeoUtils exposes **metadata properties and methods** consistently across geospatial objects.
These rely only on metadata and therefore **do not load or modify underlying data arrays**.

```{list-table} Common API from metadata operations
:widths: 3 7
:header-rows: 1
:align: left
:class: tight-table

* - Attribute / Method
  - Description

* - <span class="gu-table-section">Raster / Vector / Point</span>
  -

* - {attr}`~RasterBase.crs`
  - Coordinate reference system (CRS) of object.

* - {attr}`~RasterBase.bbox`
  - Bounding box of object.

* - {attr}`~RasterBase.footprint`
  - Footprint polygon geometry of object.

* - {attr}`~RasterBase.is_loaded`
  - Whether geospatial object is loaded in-memory.

* - {attr}`~RasterBase.name`
  - Filename of object on disk, if it exists.

* - {meth}`get_bbox_projected() <RasterBase.get_bbox_projected>`
  - Bounding box projected in another CRS.

* - {meth}`get_footprint_projected() <RasterBase.get_footprint_projected>`
  - Footprint polygon geometry in other CRS.

* - {meth}`get_metric_crs() <RasterBase.get_metric_crs>`
  - Get metric CRS suitable for this object.

* - {meth}`~RasterBase.info`
  - Summary of attributes for geospatial object.

* - <span class="gu-table-section">Raster / Point</span>
  -

* - {attr}`~RasterBase.data`
  - Data array (2D or 3D for raster, 1D for point cloud).

* - {attr}`~RasterBase.shape`
  - Shape of data array.

* - {attr}`~RasterBase.is_mask`
  - Whether object is a mask.

* - <span class="gu-table-section">Raster</span>
  -

* - {attr}`~RasterBase.transform`
  - Geotransform to map raster cells to spatial coordinates.

* - {attr}`~RasterBase.nodata`
  - Nodata value used to represent missing data on disk.

* - {attr}`~RasterBase.area_or_point`
  - Interpretation of raster cell values, area-average or point-center.

* - <span class="gu-table-section">Point</span>
  -

* - {attr}`~geoutils.pointcloud.base.GeometryPointCloudBase.point_count`
  - Number of points in the point cloud.
```
