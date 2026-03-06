(method-summary)=

# Feature overview

GeoUtils provides a unified API for manipulating **raster**, **vector**, and **point-cloud** data, with **scalable execution** for most raster operations.

The **tables below** summarize the core operations of GeoUtils, their scalability and backends.

If you are interested in converting from GDAL/OGR, see our {ref}`cheatsheet-osgeo` page.

## Summary of methods and scalability

Methods of GeoUtils are shared across object types and expose a **consistent API** for clarity (similarly as the recent [GDAL CLI overhaul](https://gdal.org/en/stable/programs/index.html)). They also support convenient inputs such as **match-reference arguments** (e.g., matching a grid for reprojection or rasterization, matching bounds for cropping, matching point coordinates for interpolation). See the {ref}`core-match-ref` page for details.

Nearly all **raster operations** support **scalable execution** through [Dask](https://www.dask.org/) or Multiprocessing, allowing large datasets to be processed chunk-by-chunk without loading the full array into memory.
While the table below provide a scalability summary, details on exact **supported operations** relative to inputs/outputs are available on the {ref}`scalability-support` section.

Some operations also support multiple computational **backends** (for example SciPy or Numba implementations for numerical routines).

All methods are tested to ensure they produce **identical results** whether executed in-memory, using chunked processing, or through alternative computational backends.

## Data operations

We first describe GeoUtils' core **data operations**, which operate on underlying arrays or geometries and therefore benefit from **scalable execution**.

**Legend:** **“/”** indicates methods **shared across object types**, while **“⟷”** indicates methods **interfacing between two object types**.

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

* - {meth}`~geoutils.Raster.reproject()`
  - Reproject to other CRS. Default tolerance parameters ensure chunk-invariance.
  - ✅
  - Rasterio / PyProj

* - {meth}`~geoutils.Raster.crop()`
  - Crop to bounds. For vectors, can return geometries intersecting (untouched) or clipped.
  - ✅
  - Rasterio / GeoPandas

* - {meth}`~geoutils.Raster.translate()`
  - Apply a grid shift to object.
  - ✅
  - NumPy / GeoPandas
  
* - {meth}`~geoutils.Raster.proximity()`
  - Estimate proximity distance to target values or geometries.
  - ❌
  - SciPy
  
* - {meth}`~geoutils.Raster.plot()`
  - Visualization helper.
  - ❌
  - Matplotlib

* - <span class="gu-table-section">Raster / Point</span>
  -
  -
  -
  
* - {meth}`~geoutils.Vector.create_mask()`
  - Create boolean mask of a vector geometries over raster or point.
  - ✅
  - Rasterio / GeoPandas

* - {meth}`~geoutils.Raster.get_stats()`
  - Compute statistics of valid values over a valid mask.
  - ❌
  - NumPy / SciPy
  
* - {meth}`~geoutils.Raster.subsample()`
  - Randomly sample valid values. Chunk-invariant seed ensures reproducibility.
  - ✅
  - NumPy
  
* - {meth}`~geoutils.Raster.filter()`
  - Filter over window. Fast vectorized logic with NaN support.
  - ✅
  - SciPy

* - <span class="gu-table-section">Raster ⟷ Vector</span>
  -
  -
  -

* - {meth}`~geoutils.Raster.polygonize()`
  - Convert raster regions to vector polygons. Multiple chunked strategies for performance.
  - ✅
  - Rasterio / GeoPandas

* - {meth}`~geoutils.Vector.rasterize()`
  - Burn vector geometries onto a raster grid.
  - ✅
  - Rasterio

* - <span class="gu-table-section">Raster ⟷ Point</span>
  -
  -
  -

* - {meth}`~geoutils.Raster.interp_points()`
  - Interpolate raster at point locations. Fast regular-grid logic with added NaN propagation.
  - ✅
  - SciPy

* - {meth}`~geoutils.Raster.reduce_points()`
  - Aggregate raster values around points.
  - ❌
  - NumPy

* - {meth}`~geoutils.PointCloud.grid()`
  - Grid irregular points onto a raster grid. Multiple approaches with added NaN propagation. 
  - ❌
  - SciPy

* - {meth}`~geoutils.Raster.from_pointcloud_regular()`
  - Direct conversion when points lie on a regular grid.
  - ❌
  - NumPy
  
* - {meth}`~geoutils.Raster.to_pointcloud()`
  - Conversion to point cloud.
  - ❌
  - NumPy
```

## Metadata properties and operations

In addition to data operations, GeoUtils exposes **metadata** properties and methods consistently across geospatial objects. 
These operate only on metadata and therefore **do not load or modify underlying data arrays**.

```{list-table} Common API from metadata operations
:widths: 3 7
:header-rows: 1
:align: left
:class: tight-table

* - Attribute / Method
  - Description

* - <span class="gu-table-section">Raster / Vector / Point</span>
  -

* - {attr}`~geoutils.Raster.crs`
  - Coordinate reference system (CRS) of object.

* - {attr}`~geoutils.Raster.bounds`
  - Bounding box of object.

* - {attr}`~geoutils.Raster.footprint`
  - Footprint polygon geometry of object.
  
* - {attr}`~geoutils.Raster.is_loaded`
  - Whether geospatial object is loaded in-memory.

* - {attr}`~geoutils.Raster.name`
  - Filename of object on disk, if it exists.
  
* - {meth}`~geoutils.Raster.get_bounds_projected()`
  - Bounds projected in other CRS.

* - {meth}`~geoutils.Raster.get_footprint_projected()`
  - Footprint polygon geometry in other CRS.
 
* - {meth}`~geoutils.Raster.get_metric_crs()`
  - Get metric CRS suitable for this object.
  
* - {meth}`~geoutils.Raster.info()`
  - Summary of attributes for geospatial object.
  
* - <span class="gu-table-section">Raster / Point</span>
  -

* - {attr}`~geoutils.Raster.data`
  - Data array (2D grid for raster, 1D for point cloud).

* - {attr}`~geoutils.Raster.shape`
  - Shape of data array.

* - {attr}`~geoutils.Raster.is_mask`
  - Whether object is a mask. Clarifies ambiguity of raster/point file types often not supporting boolean types.

* - <span class="gu-table-section">Raster</span>
  -

* - {attr}`~geoutils.Raster.transform`
  - Geotransform to map raster indices to spatial coordinates.

* - {attr}`~geoutils.Raster.nodata`
  - Nodata value used to represent missing data on disk.
  
* - {attr}`~geoutils.Raster.area_or_point`
  - Pixel interpretation of raster values, either center point or area average.

* - <span class="gu-table-section">Point</span>
  -

* - {attr}`~geoutils.PointCloud.point_count`
  - Number of points in the point cloud.
```



