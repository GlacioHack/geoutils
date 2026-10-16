(scalability-support)=
# Supported operations

GeoUtils supports **scalable execution for most of its operations**, nearly all for rasters and point clouds, and most for vectors. 

Chunked implementations can run through either **Dask** (via the Xarray and Pandas accessors) or **Multiprocessing** (via GeoUtils objects).

The interfaces expose shared geospatial methods, and both chunked backends reuse their numerical calculations.
Tests compare chunked and in-memory results, using numerical tolerances for floating point reductions.

## Table summary

The table below summarizes the **scalability support** of GeoUtils operations with respect to their input and output
behavior. Operations marked as chunked are also validated with inputs larger than the memory available to one worker.

If you are unfamiliar with **chunked and lazy execution** or **deferred I/O**, see the {ref}`scalability-concept` page.

```{admonition} Legend
- {bdg-success}`Chunked` — Processes in chunks, without loading (for input) and/or returning (for output) the full data. This is **also lazy (deferred execution) when using Dask**, but not when using Multiprocessing.
- {bdg-secondary}`In-memory` — Loads (for input) or returns (for output) full data in-memory.
- {bdg-primary}`Deferred I/O` — Deferred input/output by updating internal metadata (as Xarray's {meth}`~xarray.DataArray.isel`).
- {bdg-light}`N/A` — Chunking does not apply, as input or output is tiny (mostly metadata).

The **`⟶`** symbol denotes methods interfacing from one specific object type to another, with point-cloud type shortened to "Point".
The **memory usage** column lists the number of input chunks loaded in memory for a given operation.
```

```{list-table}
:name: Scalability summary
:widths: 1 1 1 2
:header-rows: 1
:align: center
:class: tight-table

* - Method
  - Input
  - Output
  - Memory usage (# chunks)

* - <span class="gu-table-section">Raster ⟶ Raster</span>
  -
  -
  -

* - {meth}`reproject() <RasterBase.reproject>`<sup><a href="#scalability-note-reprojection">1</a></sup>
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - Depends on grid change, source chunks and neighborhood
* - {meth}`crop() <RasterBase.crop>` /
    {meth}`icrop() <RasterBase.icrop>`
  - {bdg-primary}`Deferred I/O`
  - {bdg-primary}`Deferred I/O`
  - 0
* - {meth}`clip() <RasterBase.clip>`
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - ~1
* - {meth}`translate() <RasterBase.translate>`
  - {bdg-primary}`Deferred I/O`
  - {bdg-primary}`Deferred I/O`
  - 0
* - {meth}`copy() <RasterBase.copy>`
  - {bdg-primary}`Deferred I/O`
  - {bdg-primary}`Deferred I/O`
  - 0
* - {meth}`filter() <RasterBase.filter>`
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - ~2–3 (if small filter window)
* - {meth}`proximity() <RasterBase.proximity>`
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - Depends on `max_distance` and chunk size

* - {meth}`krige() <RasterBase.krige>`
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - Depends on covariance range and `max_overlap`
* - {meth}`random_field() <RasterBase.random_field>`<sup><a href="#scalability-note-random-fields">2</a></sup>
  - {bdg-light}`N/A` (measurements)<br>
    {bdg-success}`Chunked` (mask/predictors)
  - {bdg-success}`Chunked`
  - Output tile and shared random model
* - {meth}`fill_nodata() <RasterBase.fill_nodata>` /
    {meth}`sieve() <RasterBase.sieve>`
  - {bdg-secondary}`In-memory`
  - {bdg-secondary}`In-memory`
  - Complete grid per processed variable

* - <span class="gu-table-section">Raster ⟶ Point</span>
  -
  -
  -

* - {meth}`subsample() <RasterBase.subsample>` /
    {meth}`to_pointcloud() <RasterBase.to_pointcloud>`
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - ~1
* - {meth}`interp_at_points() <RasterBase.interp_at_points>`
  - {bdg-success}`Chunked`
  - {bdg-secondary}`In-memory`
  - ~1 raster chunk and 1 point partition
* - {meth}`reduce_at_points() <RasterBase.reduce_at_points>`
  - {bdg-success}`Chunked`
  - {bdg-secondary}`In-memory`
  - ~1 raster chunk and 1 point partition
* - {meth}`cosample() <RasterBase.cosample>`
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - ~1–2

* - <span class="gu-table-section">Raster ⟶ Vector</span>
  -
  -
  -

* - {meth}`polygonize() <RasterBase.polygonize>`
  - {bdg-success}`Chunked`
  - {bdg-secondary}`In-memory`
  - ~1–2

* - <span class="gu-table-section">Raster ⟶ Other</span>
  -
  -
  -

* - {meth}`plot() <RasterBase.plot>`
  - {bdg-success}`Chunked`
  - {bdg-light}`N/A`
  - ~1
* - {meth}`stats() <RasterBase.stats>`
  - {bdg-success}`Chunked`
  - {bdg-light}`N/A`
  - ~1
* - {meth}`pairsample() <RasterBase.pairsample>`
  - {bdg-success}`Chunked`
  - {bdg-secondary}`In-memory`
  - ~1 + bounded pair sample
* - {meth}`variogram() <RasterBase.variogram>`
  - {bdg-success}`Chunked`
  - {bdg-light}`N/A`
  - ~1 + bounded pair sample
* - {meth}`estimate_error_structure() <RasterBase.estimate_error_structure>`
  - {bdg-success}`Chunked`
  - {bdg-light}`N/A`
  - Group summaries and bounded pair sample
* - {meth}`stats() <RasterBase.stats>` with ``fractional=True``
  - {bdg-secondary}`In-memory`
  - {bdg-light}`N/A`
  - Complete raster values and polygon intersections

* - <span class="gu-table-section">Point ⟶ Point</span>
  -
  -
  -

* - {meth}`crop() <geoutils.pointcloud.base.GeometryPointCloudBase.crop>`
  - {bdg-primary}`Deferred I/O`
  - {bdg-primary}`Deferred I/O`
  - 0
* - {meth}`clip() <geoutils.pointcloud.base.GeometryPointCloudBase.clip>`
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - ~1
* - {meth}`reproject() <geoutils.pointcloud.base.GeometryPointCloudBase.reproject>`
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - ~1
* - {meth}`translate() <geoutils.pointcloud.base.GeometryPointCloudBase.translate>`
  - {bdg-secondary}`In-memory`
  - {bdg-secondary}`In-memory`
  - —
* - {meth}`subsample() <PointCloudBase.subsample>`
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - ~1
* - {meth}`cosample() <PointCloudBase.cosample>`
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - ~1–2
* - {meth}`filter() <PointCloudBase.filter>`<sup><a href="#scalability-note-point-neighborhoods">3</a></sup>
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - ~1 point partition and nearby points within the search radius
* - {meth}`random_field() <PointCloudBase.random_field>`<sup><a href="#scalability-note-random-fields">2</a></sup>
  - {bdg-light}`N/A` (measurements)<br>
    {bdg-success}`Chunked` (coordinates/predictors)
  - {bdg-success}`Chunked`
  - Point partition and shared random model

* - <span class="gu-table-section">Point ⟶ Raster</span>
  -
  -
  -

* - {meth}`grid() <PointCloudBase.grid>`<sup><a href="#scalability-note-point-neighborhoods">3</a></sup>
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - ~1 point partition and 1 output chunk
* - {meth}`krige() <PointCloudBase.krige>`
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - Points within covariance range and 1 output chunk
* - {meth}`from_pointcloud_regular() <RasterBase.from_pointcloud_regular>`
  - {bdg-secondary}`In-memory`
  - {bdg-secondary}`In-memory`
  - —

* - <span class="gu-table-section">Vector ⟶ Vector</span>
  -
  -
  -

* - {meth}`crop() <VectorBase.crop>`
  - {bdg-primary}`Deferred I/O`
  - {bdg-primary}`Deferred I/O`
  - 0
* - {meth}`clip() <VectorBase.clip>`
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - ~1
* - {meth}`reproject() <VectorBase.reproject>` /
    {meth}`translate() <VectorBase.translate>`
  - {bdg-secondary}`In-memory`
  - {bdg-secondary}`In-memory`
  - —

* - <span class="gu-table-section">Vector ⟶ Raster</span>
  -
  -
  -

* - {meth}`rasterize() <VectorBase.rasterize>`
  - {bdg-secondary}`In-memory`
  - {bdg-success}`Chunked`
  - ~1
* - {meth}`create_mask() <VectorBase.create_mask>`
  - {bdg-secondary}`In-memory`
  - {bdg-success}`Chunked`
  - ~1
* - {meth}`proximity() <VectorBase.proximity>`
  - {bdg-secondary}`In-memory`
  - {bdg-success}`Chunked`
  - Depends on `max_distance` and chunk size

* - <span class="gu-table-section">Vector ⟶ Point</span>
  -
  -
  -

* - {meth}`create_mask() <VectorBase.create_mask>`
  - {bdg-secondary}`In-memory`
  - {bdg-success}`Chunked` (Dask)
  - ~1 point partition + vector geometry

* - <span class="gu-table-section">Point ⟶ Other</span>
  -
  -
  -

* - {meth}`stats() <PointCloudBase.stats>`
  - {bdg-success}`Chunked`
  - {bdg-light}`N/A`
  - ~1
* - {meth}`pairsample() <PointCloudBase.pairsample>`
  - {bdg-success}`Chunked`
  - {bdg-secondary}`In-memory`
  - ~1 + spatial index
* - {meth}`variogram() <PointCloudBase.variogram>`
  - {bdg-success}`Chunked`
  - {bdg-light}`N/A`
  - ~1 + spatial index
* - {meth}`estimate_error_structure() <PointCloudBase.estimate_error_structure>`
  - {bdg-success}`Chunked`
  - {bdg-light}`N/A`
  - Group summaries and bounded pair sample
* - {meth}`plot() <geoutils.pointcloud.base.GeometryPointCloudBase.plot>`
  - {bdg-success}`Chunked`
  - {bdg-light}`N/A`
  - ~1

* - <span class="gu-table-section">Raster / Point ⟶ Uncertainty</span>
  -
  -
  -

* - {func}`propagate() <geoutils.uncertainty.propagate>`
  - {bdg-success}`Chunked` (Dask)
  - {bdg-success}`Chunked` (Dask)
  - Depends on the spatial operation
```

Note that nearly all **raster inputs/outputs** and most **point inputs/outputs** methods support {bdg-success}`Chunked`, while **vector inputs** are often {bdg-secondary}`In-memory`, as usually less limiting.

See {ref}`uncertainty` for error workflows and {ref}`scalability-formats` for reading and writing limits imposed
by the file layout.

For more insights into chunked implementation strategies and behaviour expected for each operation, see the {ref}`scalability-logic` page.

## Notes

1. <span id="scalability-note-reprojection"></span>Chunked reprojection requires an **affine source grid**;
   GCP/RPC transformation models are not supported.

2. <span id="scalability-note-random-fields"></span>Chunked random fields with GPyTorch **use an inducing-grid approximation**.

3. <span id="scalability-note-point-neighborhoods"></span>Chunked point resampling (gridding, filtering) **requires a finite neighborhood `radius`**.
