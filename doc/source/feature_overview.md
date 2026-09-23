(feature-overview)=
# Feature and scalability overview

GeoUtils provides a unified API for manipulating **raster**, **vector**, and **point cloud** data, and provides **scalable CPU execution** for most raster and point cloud operations, as well as some vector operations, through Dask and Multiprocessing.

As many of our numerical operations rely on **NumPy, SciPy or Numba**, those are planned to be linked to their **GPU** counterparts (**CuPy** and **Numba CUDA**) in the future.

The **{ref}`summary tables<tables-overview>` directly below** list the core features of GeoUtils, their scalability and available backends.

```{seealso}
If you are interested in **porting from GDAL/OGR**, see our {ref}`cheatsheet-osgeo` page.
While tables below provide a scalability summary, the detailed **scalable execution behaviour of all operations** is available on the {ref}`scalability-support` page.
For **performance comparisons**, see the {ref}`benchmarking-performance` page.
```

## Summary

GeoUtils exposes a **consistent API across raster, vector and point cloud objects** where possible (similar in spirit to the recent [GDAL CLI overhaul](https://gdal.org/en/stable/programs/index.html)). Many operations also support convenient **match-reference arguments** (e.g., matching a grid for reprojection or rasterization, bounds for cropping, or point coordinates for interpolation). See the {ref}`core-match-ref` page for details.

At its core, GeoUtils provides two interchangeable ways to work with geospatial data, exposing **identical APIs**:

- **Accessors** that extend existing data structures ({class}`rst <geoutils.RasterAccessor>` for **rasters** with **Xarray**, {class}`pc <geoutils.PointCloudAccessor>` and {class}`vct <geoutils.VectorAccessor>` for **point clouds** and **vectors** with **GeoPandas**),
- **GeoUtils objects** {class}`~geoutils.Raster`, {class}`~geoutils.PointCloud`, {class}`~geoutils.Vector`.

All **raster** and **point cloud** operations support **scalable execution** using [Dask](https://www.dask.org/) or Multiprocessing, allowing large datasets to be processed **chunk-by-chunk without loading the full array into memory**. Some **vector operations** also support scalable execution, although vector inputs are often less limiting.

Additionally, some numerical routines of GeoUtils provide multiple computational **backends** (e.g., SciPy or Numba implementations).

All methods are tested to ensure they produce **identical results** whether executed **in-memory**, **chunked**, or with **different computational backends**.

(tables-overview)=
## Data operations

We first describe GeoUtils' core **data operations**, which operate on underlying arrays or geometries and can therefore benefit from **scalable execution**.

```{admonition} Legend
- **`/`** denotes methods **shared across object types**.
- **`⟷`** denotes methods **interfacing between two object types**.
```

```{list-table} Common API for data operations
:widths: 3 5 1 2
:header-rows: 1
:align: left
:class: tight-table

* - Method
  - Notes
  - Scalable
  - Backend

* - <span class="gu-table-section">Raster / Vector / Point</span>
  -
  -
  -

* - {meth}`reproject() <RasterBase.reproject>`
  - Reproject to other CRS. Also resamples to new grid for rasters, with default parameters ensuring chunk-invariance.
  - ✅ (raster/point)
  - Rasterio / PyProj

* - {meth}`crop() <RasterBase.crop>`
  - Crop to a bounding box without changing values or geometries (deferred I/O). Vectors are kept either by
    intersection or containment.
  - ✅
  - Rasterio / GeoPandas

* - {meth}`clip() <RasterBase.clip>`
  - Clip to an exact geometry: mask cells for rasters, remove data for points, and cut geometries for vectors.
  - ✅
  - Rasterio / GeoPandas

* - {meth}`translate() <RasterBase.translate>`
  - Apply a grid shift to object.
  - ✅ (raster)
  - NumPy / GeoPandas

* - {meth}`plot() <RasterBase.plot>`
  - Visualization helper.
  - ✅ (raster/point)
  - Matplotlib

* - <span class="gu-table-section">Raster / Point</span>
  -
  -
  -

* - {meth}`create_mask() <VectorBase.create_mask>`
  - Create a boolean mask from vector geometries over a raster or point cloud.
  - ✅ (raster output)
  - Rasterio / GeoPandas

* - {meth}`stats() <RasterBase.stats>`
  - Compute statistics of valid values over a valid mask.
  - ✅
  - NumPy / SciPy

* - {meth}`stats() <RasterBase.stats>` with ``by``
  - Compute statistics by continuous bins, discrete categories or vector geometries (zonal statistics).
  - ✅
  - Pandas / NumPy / Dask

* - {meth}`subsample() <RasterBase.subsample>`
  - Randomly sample valid raster cells as a point cloud or value/index array.
  - ✅
  - NumPy

* - {meth}`cosample() <RasterBase.cosample>`
  - Select matching finite values from two datasets. Returns a raster or point cloud on the chosen support.
  - ✅
  - NumPy / Dask

* - {meth}`pairsample() <RasterBase.pairsample>`
  - Select finite pairs across spatial distances. Returns a compact pair dataset.
  - ✅
  - NumPy / SciPy / Dask

* - {meth}`variogram() <RasterBase.variogram>`
  - Estimate and fit semivariance by distance from sampled pairs. Returns lag statistics and a model.
  - ✅
  - NumPy / SciKit-GStat

* - {meth}`filter() <RasterBase.filter>`
  - Filter over window. Fast vectorized logic with NaN support.
  - ✅
  - SciPy

* - {meth}`proximity() <RasterBase.proximity>`
  - Estimate proximity distance to target values.
  - ✅ (raster)
  - SciPy


* - <span class="gu-table-section">Raster ⟷ Vector</span>
  -
  -
  -

* - {meth}`polygonize() <RasterBase.polygonize>`
  - Convert raster regions to vector polygons. Multiple chunked strategies for performance.
  - ✅
  - Rasterio / GeoPandas

* - {meth}`rasterize() <VectorBase.rasterize>`
  - Burn vector geometries onto a raster grid.
  - ✅
  - Rasterio

* - <span class="gu-table-section">Raster ⟷ Point</span>
  -
  -
  -

* - {meth}`interp_points() <RasterBase.interp_points>`
  - Interpolate raster at point locations. Fast regular-grid logic with added NaN propagation.
  - ✅
  - SciPy

* - {meth}`reduce_points() <RasterBase.reduce_points>`
  - Aggregate raster values around points.
  - ❌
  - NumPy

* - {meth}`grid() <PointCloudBase.grid>`
  - Grid irregular points onto a raster grid. Multiple approaches with added NaN propagation.
  - ✅
  - SciPy

* - {meth}`from_pointcloud_regular() <RasterBase.from_pointcloud_regular>`
  - Direct conversion when points lie on a regular grid.
  - ❌
  - NumPy

* - {meth}`to_pointcloud() <RasterBase.to_pointcloud>`
  - Conversion to point cloud.
  - ✅
  - NumPy
```

## Metadata properties and operations

In addition to data operations, GeoUtils exposes **metadata** properties and methods consistently across geospatial objects.
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

* - {meth}`info() <geoutils.Raster.info>`
  - Summary of attributes for geospatial object.

* - <span class="gu-table-section">Raster / Point</span>
  -

* - {attr}`~RasterBase.data`
  - Data array (2D or 3D for raster, 1D for point cloud).

* - {attr}`~RasterBase.shape`
  - Shape of data array.

* - {attr}`~RasterBase.is_mask`
  - Whether object is a mask. Clarifies ambiguity of raster/point file types often not supporting boolean types.

* - <span class="gu-table-section">Raster</span>
  -

* - {attr}`~RasterBase.transform`
  - Geotransform to map raster cells to spatial coordinates.

* - {attr}`~RasterBase.nodata`
  - Nodata value used to represent missing data on disk.

* - {attr}`~RasterBase.area_or_point`
  - Interpretation of raster cell values, either an area-average or point-center.

* - <span class="gu-table-section">Point</span>
  -

* - {attr}`~PointCloudBase.point_count`
  - Number of points in the point cloud.
```
