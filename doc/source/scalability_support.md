(scalability-support)=
# Supported operations

GeoUtils supports **scalable execution for most of its raster and point cloud methods**, including nearly all **raster–point** and **raster–vector** interface operations. Support for **vector** accessors is partially supported through **Dask-GeoPandas**.

Chunked implementations can run through either **Dask** (via the Xarray and Pandas accessors) or **Multiprocessing** (via GeoUtils objects such as {class}`~geoutils.Raster`).

Both object types (accessors or GeoUtils) expose the **exact same API**, and both chunked backends use the same internal logic, and all methods are tested to **yield identical output** as in-memory.

## Table summary

The table below summarizes the **scalability support** of GeoUtils operations with respect to their input and output
behavior. Operations marked as chunked are also validated with inputs larger than the memory available to one worker.

If you are unfamiliar with **chunked and lazy execution** or **deferred I/O**, see the {ref}`scalability-concept` page.

```{admonition} Legend
- {bdg-success}`Chunked` — Processes in chunks, without loading (for input) and/or returning (for output) the full data. This is **also lazy (deferred execution) when using Dask**, but not when using Multiprocessing.
- {bdg-secondary}`In-memory` — Loads (for input) or returns (for output) full data in-memory.
- {bdg-primary}`Deferred I/O` — Deferred input/output by updating internal metadata (as Xarray's {meth}`~xarray.DataArray.isel`).

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

* - {meth}`reproject() <RasterBase.reproject>`
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - ~4 (with default output chunking)
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

* - <span class="gu-table-section">Raster ⟶ Point</span>
  -
  -
  -

* - {meth}`subsample() <RasterBase.subsample>` /
    {meth}`to_pointcloud() <RasterBase.to_pointcloud>`
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - ~1
* - {meth}`interp_points() <RasterBase.interp_points>`
  - {bdg-success}`Chunked`
  - {bdg-secondary}`In-memory`
  - ~1 raster chunk and 1 point partition
* - {meth}`reduce_points() <RasterBase.reduce_points>`
  - {bdg-secondary}`In-memory`
  - {bdg-secondary}`In-memory`
  - —
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
  - {bdg-secondary}`In-memory`
  - ~1
* - {meth}`stats() <RasterBase.stats>`
  - {bdg-success}`Chunked`
  - {bdg-secondary}`In-memory`
  - ~1
* - {meth}`pairsample() <RasterBase.pairsample>` /
    {meth}`variogram() <RasterBase.variogram>`
  - {bdg-success}`Chunked`
  - {bdg-secondary}`In-memory`
  - ~1 + bounded pair sample

* - <span class="gu-table-section">Point ⟶ Point</span>
  -
  -
  -

* - {meth}`crop() <PointCloudBase.crop>`
  - {bdg-primary}`Deferred I/O`
  - {bdg-primary}`Deferred I/O`
  - 0
* - {meth}`clip() <PointCloudBase.clip>`
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - ~1
* - {meth}`reproject() <PointCloudBase.reproject>`
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - ~1
* - {meth}`translate() <PointCloudBase.translate>`
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

* - <span class="gu-table-section">Point ⟶ Raster</span>
  -
  -
  -

* - {meth}`grid() <PointCloudBase.grid>`
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - ~1 point partition and 1 output chunk
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
  - {bdg-secondary}`In-memory`
  - —

* - <span class="gu-table-section">Point ⟶ Other</span>
  -
  -
  -

* - {meth}`stats() <PointCloudBase.stats>`
  - {bdg-success}`Chunked`
  - {bdg-secondary}`In-memory`
  - ~1
* - {meth}`pairsample() <PointCloudBase.pairsample>` /
    {meth}`variogram() <PointCloudBase.variogram>`
  - {bdg-success}`Chunked`
  - {bdg-secondary}`In-memory`
  - ~1 + spatial index
* - {meth}`plot() <PointCloudBase.plot>`
  - {bdg-success}`Chunked`
  - {bdg-secondary}`In-memory`
  - ~1
```

Note that nearly all **raster inputs/outputs** and most **point inputs/outputs** methods support {bdg-success}`Chunked`, while **vector inputs** are often {bdg-secondary}`In-memory`, as usually less limiting.

For more insights into chunked implementation strategies and behaviour expected for each operation, see the {ref}`scalability-logic` page.
