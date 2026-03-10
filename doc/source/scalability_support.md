(scalability-support)=
# Supported operations

GeoUtils supports **scalable execution for most of its raster methods**, including nearly all **raster–point** and **raster–vector** interface operations. Support for **point-cloud methods** is partially supported and under development, while **vector methods** may also gain scalable support in the future (lower priority).

Chunked implementations can run through either **Dask** (via the Xarray and Pandas accessors) or **Multiprocessing** (via GeoUtils objects such as {class}`~geoutils.Raster`).

Both object types (accessors or GeoUtils) expose the **exact same API**, and both chunked backends use the same internal logic, and all methods are tested to **yield identical output** as in-memory.

## Table summary

The table below summarizes the **scalability support** of GeoUtils operations with respect to their input and output behavior.

If you are unfamiliar with **chunked and lazy execution** or **deferred I/O**, see the {ref}`scalability-concept` page. 

**Legend:**
- {bdg-success}`Chunked` —  Processes in chunks, without loading (for input) and/or returning (for output) the full data. This is **also lazy (deferred execution) when using Dask**, but not when using Multiprocessing.
- {bdg-secondary}`In-memory` — Loads (for input) or returns (for output) full data in-memory.
- {bdg-primary}`Deferred I/O` — Deferred input/output by updating internal metadata (as Xarray's {meth}`~xarray.DataArray.isel`).

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

* - {meth}`~geoutils.Raster.reproject`
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - ~4 (default), or ~downsampling²
* - {meth}`~geoutils.Raster.crop` / {meth}`~geoutils.Raster.icrop`
  - {bdg-primary}`Deferred I/O`
  - {bdg-primary}`Deferred I/O`
  - 0
* - {meth}`~geoutils.Raster.translate`
  - {bdg-primary}`Deferred I/O`
  - {bdg-primary}`Deferred I/O`
  - 0
* - {meth}`~geoutils.Raster.copy`
  - {bdg-primary}`Deferred I/O`
  - {bdg-primary}`Deferred I/O`
  - 0
* - {meth}`~geoutils.Raster.filter`
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - ~2–3 (if small filter window)
* - {meth}`~geoutils.Raster.proximity`
  - {bdg-secondary}`In-memory`
  - {bdg-secondary}`In-memory`
  - —

* - <span class="gu-table-section">Raster ⟶ Point</span>
  -
  -
  -

* - {meth}`~geoutils.Raster.subsample`
  - {bdg-success}`Chunked`
  - {bdg-secondary}`In-memory`
  - ~1 
* - {meth}`~geoutils.Raster.interp_points`
  - {bdg-success}`Chunked`
  - {bdg-secondary}`In-memory`
  - ~1
* - {meth}`~geoutils.Raster.reduce_points`
  - {bdg-secondary}`In-memory`
  - {bdg-secondary}`In-memory`
  - —

* - <span class="gu-table-section">Raster ⟶ Vector</span>
  -
  -
  -

* - {meth}`~geoutils.Raster.polygonize`
  - {bdg-success}`Chunked`
  - {bdg-secondary}`In-memory`
  - ~1–2

* - <span class="gu-table-section">Raster ⟶ Other</span>
  -
  -
  -

* - {meth}`~geoutils.Raster.plot`
  - {bdg-secondary}`In-memory`
  - {bdg-secondary}`In-memory`
  - —
* - {meth}`~geoutils.Raster.get_stats`
  - {bdg-secondary}`In-memory`
  - {bdg-secondary}`In-memory`
  - —

* - <span class="gu-table-section">Point ⟶ Point</span>
  -
  -
  -

* - {meth}`~geoutils.PointCloud.reproject`
  - {bdg-secondary}`In-memory`
  - {bdg-secondary}`In-memory`
  - —
* - {meth}`~geoutils.PointCloud.translate`
  - {bdg-secondary}`In-memory`
  - {bdg-secondary}`In-memory`
  - —
* - {meth}`~geoutils.PointCloud.crop`
  - {bdg-secondary}`In-memory`
  - {bdg-secondary}`In-memory`
  - —

* - <span class="gu-table-section">Point ⟶ Raster</span>
  -
  -
  -

* - {meth}`~geoutils.PointCloud.grid`
  - {bdg-secondary}`In-memory`
  - {bdg-secondary}`In-memory`
  - —
* - {meth}`~geoutils.Raster.from_pointcloud_regular`
  - {bdg-secondary}`In-memory`
  - {bdg-secondary}`In-memory`
  - —

* - <span class="gu-table-section">Vector ⟶ Raster</span>
  -
  -
  -

* - {meth}`~geoutils.Vector.rasterize`
  - {bdg-secondary}`In-memory`
  - {bdg-success}`Chunked`
  - ~1
* - {meth}`~geoutils.Vector.create_mask`
  - {bdg-secondary}`In-memory`
  - {bdg-success}`Chunked`
  - ~1

* - <span class="gu-table-section">Vector ⟶ Point</span>
  -
  -
  -

* - {meth}`~geoutils.Vector.create_mask`
  - {bdg-secondary}`In-memory`
  - {bdg-secondary}`In-memory`
  - —

* - <span class="gu-table-section">Point ⟶ Other</span>
  -
  -
  -

* - {meth}`~geoutils.PointCloud.get_stats`
  - {bdg-secondary}`In-memory`
  - {bdg-secondary}`In-memory`
  - —
```

Note that nearly all **raster inputs/outputs** methods support {bdg-success}`Chunked`, while **point and vector inputs/outputs** are currently {bdg-secondary}`In-memory`, as often less limiting.

For more insights into chunked implementation strategies and behaviour expected for each operation, see the {ref}`scalability-logic` page.