(scalability-support)=
# Supported operations

GeoUtils supports **scalable execution for most of its operations**, nearly all for rasters and point clouds, and most for vectors.

Chunked implementations can run through either **Dask** (via the Xarray and Pandas accessors) or **Multiprocessing** (via GeoUtils objects).

The two interfaces expose the **same operations**, and chunked implementations yield **identical results** as in-memory operations, including with different backends (e.g., SciPy, Numba).

## Table summary

The table below summarizes the **scalability support** of GeoUtils operations with respect to their input and output
behavior.

If you are unfamiliar with **chunked and lazy execution** or **deferred I/O**, see {ref}`scalability-concept`.

```{admonition} Legend
- {bdg-success}`Chunked` — Processes in chunks, without loading (for input) and/or returning (for output) the full data. This is **also lazy (deferred execution) when using Dask**.
- {bdg-secondary}`In-memory` — Loads (for input) or returns (for output) full data in-memory.
- {bdg-primary}`Deferred I/O` — Deferred input/output by updating internal metadata (as Xarray's {meth}`~xarray.DataArray.isel`).
- {bdg-light}`N/A` — Chunking does not apply, as input or output is tiny (mostly metadata).

The **`⟶`** symbol denotes methods interfacing from one specific object type to another, with point-cloud type shortened to "Point".

**Memory usage** estimates the number of chunks in-memory: **1R**/**1P**/**1V** mean one
raster/point/vector chunk respectively, **R/P** means either raster or point. See bottom notes for special cases.
```

```{list-table}
:name: Scalability summary
:widths: 2 1 1 1
:header-rows: 1
:align: center
:class: tight-table

* - Method
  - Input
  - Output
  - Memory usage

* - <span class="gu-table-section">Raster ⟶ Raster</span>
  -
  -
  -

* - {meth}`reproject() <RasterBase.reproject>`
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - 1R + CRS overlap<sup><a href="#scalability-note-reprojection">1</a></sup>
* - {meth}`crop() <RasterBase.crop>` /
    {meth}`icrop() <RasterBase.icrop>`
  - {bdg-primary}`Deferred I/O`
  - {bdg-primary}`Deferred I/O`
  - 0
* - {meth}`clip() <RasterBase.clip>`<sup><a href="#scalability-note-preliminary-reads">5</a></sup>
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - 1R
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
  - 1R + overlap<sup><a href="#scalability-note-neighborhoods">2</a></sup>
* - {meth}`proximity() <RasterBase.proximity>`
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - 1R + overlap<sup><a href="#scalability-note-neighborhoods">2</a></sup>

* - {meth}`krige() <RasterBase.krige>`
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - 1R + overlap<sup><a href="#scalability-note-neighborhoods">2</a></sup>
* - {meth}`random_field() <RasterBase.random_field>`
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - 1R + model<sup><a href="#scalability-note-random-fields">3</a></sup>
* - {meth}`fill_nodata() <RasterBase.fill_nodata>` /
    {meth}`sieve() <RasterBase.sieve>`
  - {bdg-secondary}`In-memory`
  - {bdg-secondary}`In-memory`
  - —

* - <span class="gu-table-section">Raster ⟶ Point</span>
  -
  -
  -

* - {meth}`subsample() <RasterBase.subsample>`<sup><a href="#scalability-note-preliminary-reads">5</a></sup> /
    {meth}`to_pointcloud() <RasterBase.to_pointcloud>`<sup><a href="#scalability-note-preliminary-reads">5</a></sup>
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - 1R + 1P
* - {meth}`interp_at_points() <RasterBase.interp_at_points>`
  - {bdg-success}`Chunked`
  - {bdg-secondary}`In-memory`
  - 1R + 1P + overlap<sup><a href="#scalability-note-neighborhoods">2</a></sup>
* - {meth}`reduce_at_points() <RasterBase.reduce_at_points>`
  - {bdg-success}`Chunked`
  - {bdg-secondary}`In-memory`
  - 1R + 1P + overlap<sup><a href="#scalability-note-neighborhoods">2</a></sup>
* - {meth}`cosample() <RasterBase.cosample>`<sup><a href="#scalability-note-preliminary-reads">5</a></sup>
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - 1–2R/P

* - <span class="gu-table-section">Raster ⟶ Vector</span>
  -
  -
  -

* - {meth}`polygonize() <RasterBase.polygonize>`
  - {bdg-success}`Chunked`
  - {bdg-secondary}`In-memory`
  - 1–2R

* - <span class="gu-table-section">Raster ⟶ Other</span>
  -
  -
  -

* - {meth}`plot() <RasterBase.plot>`
  - {bdg-success}`Chunked`
  - {bdg-light}`N/A`
  - 1R
* - {meth}`stats() <RasterBase.stats>`
  - {bdg-success}`Chunked`
  - {bdg-light}`N/A`
  - 1R
* - {meth}`pairsample() <RasterBase.pairsample>`
  - {bdg-success}`Chunked`
  - {bdg-secondary}`In-memory`
  - 1R + sample<sup><a href="#scalability-note-sampling-memory">4</a></sup>
* - {meth}`variogram() <RasterBase.variogram>`
  - {bdg-success}`Chunked`
  - {bdg-light}`N/A`
  - 1R + sample<sup><a href="#scalability-note-sampling-memory">4</a></sup>
* - {meth}`estimate_error_structure() <RasterBase.estimate_error_structure>`
  - {bdg-success}`Chunked`
  - {bdg-light}`N/A`
  - 1R + sample<sup><a href="#scalability-note-sampling-memory">4</a></sup>
* - {meth}`stats() <RasterBase.stats>` with ``fractional=True``
  - {bdg-secondary}`In-memory`
  - {bdg-light}`N/A`
  - —

* - <span class="gu-table-section">Point ⟶ Point</span>
  -
  -
  -

* - {meth}`crop() <geoutils.pointcloud.base.GeometryPointCloudBase.crop>`<sup><a href="#scalability-note-preliminary-reads">5</a></sup>
  - {bdg-primary}`Deferred I/O`
  - {bdg-primary}`Deferred I/O`
  - 0
* - {meth}`clip() <geoutils.pointcloud.base.GeometryPointCloudBase.clip>`<sup><a href="#scalability-note-preliminary-reads">5</a></sup>
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - 1P
* - {meth}`reproject() <geoutils.pointcloud.base.GeometryPointCloudBase.reproject>`
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - 1P
* - {meth}`translate() <geoutils.pointcloud.base.GeometryPointCloudBase.translate>`
  - {bdg-secondary}`In-memory`
  - {bdg-secondary}`In-memory`
  - —
* - {meth}`subsample() <PointCloudBase.subsample>`<sup><a href="#scalability-note-preliminary-reads">5</a></sup>
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - 1P
* - {meth}`cosample() <PointCloudBase.cosample>`<sup><a href="#scalability-note-preliminary-reads">5</a></sup>
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - 1–2R/P
* - {meth}`filter() <PointCloudBase.filter>`
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - 1P + overlap<sup><a href="#scalability-note-neighborhoods">2</a></sup>
* - {meth}`random_field() <PointCloudBase.random_field>`<sup><a href="#scalability-note-preliminary-reads">5</a></sup>
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - 1P + model<sup><a href="#scalability-note-random-fields">3</a></sup>

* - <span class="gu-table-section">Point ⟶ Raster</span>
  -
  -
  -

* - {meth}`grid() <PointCloudBase.grid>`<sup><a href="#scalability-note-preliminary-reads">5</a></sup>
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - 1P + 1R + overlap<sup><a href="#scalability-note-neighborhoods">2</a></sup>
* - {meth}`krige() <PointCloudBase.krige>`<sup><a href="#scalability-note-preliminary-reads">5</a></sup>
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - 1P + 1R + overlap<sup><a href="#scalability-note-neighborhoods">2</a></sup>
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
* - {meth}`clip() <VectorBase.clip>`<sup><a href="#scalability-note-preliminary-reads">5</a></sup>
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - 1V
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
  - 1R
* - {meth}`create_mask() <VectorBase.create_mask>`
  - {bdg-secondary}`In-memory`
  - {bdg-success}`Chunked`
  - 1R
* - {meth}`proximity() <VectorBase.proximity>`
  - {bdg-secondary}`In-memory`
  - {bdg-success}`Chunked`
  - 1R + overlap<sup><a href="#scalability-note-neighborhoods">2</a></sup>

* - <span class="gu-table-section">Vector ⟶ Point</span>
  -
  -
  -

* - {meth}`create_mask() <VectorBase.create_mask>`
  - {bdg-secondary}`In-memory`
  - {bdg-success}`Chunked`
  - 1P

* - <span class="gu-table-section">Point ⟶ Other</span>
  -
  -
  -

* - {meth}`stats() <PointCloudBase.stats>`
  - {bdg-success}`Chunked`
  - {bdg-light}`N/A`
  - 1P
* - {meth}`pairsample() <PointCloudBase.pairsample>`
  - {bdg-success}`Chunked`
  - {bdg-secondary}`In-memory`
  - 1P + index<sup><a href="#scalability-note-sampling-memory">4</a></sup>
* - {meth}`variogram() <PointCloudBase.variogram>`
  - {bdg-success}`Chunked`
  - {bdg-light}`N/A`
  - 1P + index<sup><a href="#scalability-note-sampling-memory">4</a></sup>
* - {meth}`estimate_error_structure() <PointCloudBase.estimate_error_structure>`
  - {bdg-success}`Chunked`
  - {bdg-light}`N/A`
  - 1P + index<sup><a href="#scalability-note-sampling-memory">4</a></sup>
* - {meth}`plot() <geoutils.pointcloud.base.GeometryPointCloudBase.plot>`
  - {bdg-success}`Chunked`
  - {bdg-light}`N/A`
  - 1P

* - <span class="gu-table-section">Raster / Point ⟶ Uncertainty</span>
  -
  -
  -

* - {func}`propagate() <geoutils.uncertainty.propagate>`
  - {bdg-success}`Chunked`
  - {bdg-success}`Chunked`
  - 1R/P + overlap<sup><a href="#scalability-note-neighborhoods">2</a></sup>
```

Note that nearly all **raster inputs/outputs** and most **point inputs/outputs** methods support {bdg-success}`Chunked`, while **vector inputs** are often {bdg-secondary}`In-memory`, as usually less limiting.

For more insights into chunked implementation strategies and behaviour expected for each operation, see the {ref}`scalability-logic` page.

### Notes

1. <span id="scalability-note-reprojection"></span>For reprojecion, **CRS overlap** corresponds to the source area needed for each
   output chunk after reprojection. By default, GeoUtils sets output chunks to a size that follows changes in resolution
   to avoid loading more than one source chunk + overlap. **GCP/RPC models are supported for chunked reprojection**, alongside classic affine grids.

2. <span id="scalability-note-neighborhoods"></span>The **overlap** corresponds to neighboring data needed to calculate
   values near chunk edges, which depends directly on filter window/neighborhood size. Chunked proximity, point
   filtering, gridding and kriging limit this through a maximum search distance.

3. <span id="scalability-note-random-fields"></span>The **model** is typically a small dataframe describing variable errors along predictors, and
   modelled correlation functions. All chunks use it.

4. <span id="scalability-note-sampling-memory"></span>The **sample** is the returned in-memory sample. The **index** helps
   find nearby points and uses more memory for larger point clouds.

5. <span id="scalability-note-preliminary-reads"></span>Some operations require **preliminary reads** even though their
   **output stays lazy**, see {ref}`scalability-preliminary-reads` directly below.

(scalability-preliminary-reads)=
## Preliminary reads

Some operations **need to read part of an object to clearly prepare** a chunked operation.

This happens when metadata alone is not sufficient to define the operation, for example to estimate **valid data counts** before subsampling rasters or
point clouds, or **estimate average height** in a chunk before a 3D affine transformation for elevation data.

Although the **output computation stays lazy**, these reads are performed during the function call, creating some small initial
processing time (usually minimal, a fast pass).

The affected operations are listed below:

```{list-table}
:widths: 40 60
:width: 100%
:header-rows: 1

* - Operation
  - Information read during the call
* - {meth}`subsample() <RasterBase.subsample>`, {meth}`cosample() <RasterBase.cosample>`, {meth}`pairsample() <RasterBase.pairsample>`, or using `subsample` on an operation)
  - Count valid observations and select sample indices
* - {meth}`pc.to_xarray() <geoutils.GeoDataFramePointCloudAccessor.to_xarray>`
  - Count rows when partition lengths are unknown
* - {meth}`pc.crop() <geoutils.pointcloud.base.GeometryPointCloudBase.crop>`, {meth}`pc.clip() <geoutils.pointcloud.base.GeometryPointCloudBase.clip>`
  - Select point rows to define the output size
* - {meth}`clip(mask=...) <RasterBase.clip>`
  - Read lazy mask geometries
* - {meth}`pc.grid() <PointCloudBase.grid>`, {meth}`pc.krige() <PointCloudBase.krige>`, {meth}`pc.random_field() <PointCloudBase.random_field>`
  - Determine point extents when needed
```

(scalability-computation-breakpoints)=
## Computation breakpoints

Operations with **in-memory outputs force computation** during the call, forming a **breakpoint in a lazy workflow** (though the Dask input stays lazy).

Usually, these operations **usually return small outputs** (e.g. statistics dictionary, empirical variogram).

Here is the list of operations triggering a breakpoint:

```{list-table}
:widths: 40 60
:width: 100%
:header-rows: 1

* - Operation
  - In-memory result
* - {meth}`rst.polygonize() <RasterBase.polygonize>`
  - Vector geometries
* - {meth}`stats() <RasterBase.stats>`, {meth}`variogram() <RasterBase.variogram>`, {meth}`estimate_error_structure() <RasterBase.estimate_error_structure>`
  - Statistics, variograms and error models
* - {attr}`pc.bbox <geoutils.DataArrayPointCloudAccessor.bbox>`, {attr}`vct.bbox <VectorBase.bbox>`
  - Bounding box (when metadata is unavailable)
* - {meth}`raster_equal() <RasterBase.raster_equal>`, {meth}`pointcloud_equal() <geoutils.pointcloud.base.GeometryPointCloudBase.pointcloud_equal>`, {meth}`vector_equal() <geoutils.GeoDataFrameVectorAccessor.vector_equal>`
  - Comparison results
* - {meth}`to_geoutils() <geoutils.GeoDataFramePointCloudAccessor.to_geoutils>`
  - GeoUtils objects
```

**Plotting and file writing** also naturally trigger computation, though raster writing with `compute=False` can remain deferred.
