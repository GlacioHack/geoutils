(scalability-logic)=
# Implementation strategies

Implementing **chunked execution** requires developing substantial internal logic often invisible to the user, making it difficult to understand what is happening in the background and how to potentially address a scalability issue.

Additionally, for certain methods, there is often no single best solution. We therefore propose several **strategies** to further improve performance based on the nature of the input data.

Below, we detail the logic behind our **chunked execution** implementations, both as an educational resource and to help optimize your code (such as memory usage).

## Summary

Several operations are easy to support for **chunked execution** as they directly reuse **Dask** methods:
- The {meth}`rst.filter() <RasterBase.filter>` function uses {func}`~dask.array.map_overlap` with a `depth` (overlap) half the `size` of the filter,
- The {meth}`rst.proximity() <RasterBase.proximity>` function uses {func}`~dask.array.map_overlap` with a `max_distance` parameter,
- The {meth}`vct.crop() <VectorBase.crop>`, {meth}`vct.clip() <VectorBase.clip>`, {meth}`vct.reproject() <VectorBase.reproject>` and {meth}`vct.translate() <VectorBase.translate>` functions use Dask-GeoPandas methods or {meth}`~dask.dataframe.DataFrame.map_partitions` to process GeoPandas partitions independently,
- The {meth}`pc.crop() <geoutils.pointcloud.base.GeometryPointCloudBase.crop>`, {meth}`pc.clip() <geoutils.pointcloud.base.GeometryPointCloudBase.clip>`, {meth}`pc.reproject() <geoutils.pointcloud.base.GeometryPointCloudBase.reproject>` and {meth}`pc.translate() <geoutils.pointcloud.base.GeometryPointCloudBase.translate>` functions apply the same operations to point partitions.

Other operations are more complex, requiring {ref}`specific logic described further below<specific-logic>`, in short:
- The {meth}`rst.reproject() <RasterBase.reproject>` function **maps the intersection of projected source grid chunks for each destination chunk** (with potentially different CRS, resolution and bounds),
- The {meth}`rst.polygonize() <RasterBase.polygonize>` function implements **three chunk-boundary reconciliation strategies** with different considerations for connected-component labeling and stitching of geometries,
- The {meth}`vct.rasterize() <VectorBase.rasterize>` function performs a geometry subsetting then directly **maps rasterized output blocks** only utilizing these subset geometries,
- The {meth}`rst.resample_at_points() <RasterBase.resample_at_points>` function performs a **fast equal-grid mapping of point locations in raster chunks**, expanding raster chunks by a few pixels depending on the resampling method, then performing ordered concatenation of outputs,
- The {meth}`pc.grid() <PointCloudBase.grid>` function **maps point partitions to independent output raster chunks**,
- The {meth}`pc.filter() <PointCloudBase.filter>` function **maps each target partition to nearby points across source partitions**, expanding its bounds by the search radius before reducing point neighborhoods independently,
- The {meth}`rst/pc.subsample() <RasterBase.subsample>` functions **counts valid values** per chunk, then uses **deterministic keys (top-k)** to select a random sample,
- The {meth}`rst/pc.pairsample() <RasterBase.pairsample>` functions **build candidate pairs in bounded batches**, reusing first endpoints from a few raster chunks and applying offsets to find second endpoints (multiple strategies), then concatenates accepted pairs,
- The {meth}`rst/pc.random_field() <RasterBase.random_field>` functions **evaluate one shared random model at the coordinates of each chunk**, using common global seeds.

Many operations then rely on others for their core chunked support:
- The {meth}`vct.create_mask() <VectorBase.create_mask>` function relies on chunked {meth}`vct.rasterize() <VectorBase.rasterize>`,
- The {meth}`rst.clip() <RasterBase.clip>` function relies on chunked {meth}`vct.create_mask() <VectorBase.create_mask>`, then **applies the mask independently to each raster chunk**,
- The {meth}`vct.proximity() <VectorBase.proximity>` function combines chunked {meth}`vct.create_mask() <VectorBase.create_mask>` with the overlap calculation of {meth}`rst.proximity() <RasterBase.proximity>`,
- The {meth}`rst.interp_at_points() <RasterBase.interp_at_points>` and {meth}`rst.reduce_at_points() <RasterBase.reduce_at_points>` functions use the same **mapping of points to raster chunks** as {meth}`rst.resample_at_points() <RasterBase.resample_at_points>`, with overlap covering the interpolation or reduction neighborhood,
- The {meth}`rst.to_pointcloud() <RasterBase.to_pointcloud>` function relies on chunked {meth}`rst.subsample() <RasterBase.subsample>`, assembling selected cell coordinates and values into point rows per chunk,
- The {meth}`rst/pc.cosample() <RasterBase.cosample>` function then relies on chunked {meth}`vct.create_mask() <VectorBase.create_mask>`, {meth}`rst.reproject() <RasterBase.reproject>` and {meth}`rst/pc.subsample() <RasterBase.subsample>`, with some additional chunked indexing logic,
- The {meth}`rst/pc.variogram() <RasterBase.variogram>` function relies on chunked {meth}`rst/pc.pairsample() <RasterBase.pairsample>`, followed by **chunked reduction per distance bin of variogram estimators** relying partially on Flox,
- The {meth}`rst.stack() <RasterBase.stack>` and {meth}`rst.merge() <RasterBase.merge>` functions rely on chunked {meth}`rst.reproject() <RasterBase.reproject>`, followed by simple chunked concatenation and reduction with direct Dask support,
- The {meth}`rst/pc.stats() <RasterBase.stats>` function relies on chunked {meth}`rst/pc.cosample() <RasterBase.cosample>`, fractional {meth}`vct.rasterize() <VectorBase.rasterize>` to compare on the same support with fractional coverage, followed by **grouped chunked reductions** based on Flox,
- The {meth}`rst/pc.estimate_error_structure() <RasterBase.estimate_error_structure>` functions combine chunked {meth}`rst/pc.cosample() <RasterBase.cosample>`, {meth}`rst/pc.stats() <RasterBase.stats>` and {meth}`rst/pc.variogram() <RasterBase.variogram>`, with **error standardization per chunk** between magnitude and correlation estimation,
- The {meth}`rst/pc.plot() <RasterBase.plot>` functions rely on chunked {meth}`rst.reproject() <RasterBase.reproject>` or {meth}`pc.subsample() <PointCloudBase.subsample>` to **reduce data to the display size** before plotting.

(specific-logic)=
## Logic of implementations

### Chunked reprojection

**Reprojection with chunked execution** requires mapping **destination chunks**—defined on a regular grid in the output CRS and potentially with different resolution or bounds—to **projected source chunks**, which become geometrically deformed after reprojection.

The diagram below illustrates this mapping procedure, with a new CRS and a downsampling of 2:

```{eval-rst}
.. plot:: diagrams/diagram_chunked_reproject.py
    :width: 100%
```


The source mapping buffers destination bounds by **two output pixels** and includes neighboring source chunks.
Configured operator neighborhoods can require a wider source margin. The number of source chunks read together
depends on the projection, resolution change, chunk sizes and neighborhood.

With Dask, output chunks default to the source's spatial chunk sizes in pixels. A coarser output pixel covers more
source pixels, so downsampling can increase memory per output chunk. For example, doubling both pixel dimensions
makes an output chunk cover about four times the source area. Choose smaller source chunks before reprojection
when that increase would exceed available memory. Multiprocessing uses `mp_config.chunks` for the output tiles.

Chunked reprojection supports both affine grids and GCP/RPC transformation models.

### Chunked polygonization

**Polygonization with chunked execution** requires identifying raster regions that may extend across **chunk boundaries**. Because chunks are processed independently, connected regions intersecting a boundary must later be **reconciled across neighboring chunks** to produce correct polygons.

The diagram below illustrates the three strategies implemented in GeoUtils for performing this reconciliation:

```{eval-rst}
.. plot:: diagrams/diagram_chunked_polygonize.py
    :width: 100%
```

All strategies begin by processing individual raster chunks independently, then reconstruct continuous polygons that span chunk boundaries.

Conceptually, the methods differ in how cross-chunk regions are reconstructed:
- **`label_union`** labels values in each chunk, then finds the **union of matching labels across chunk seams** before polygonization, avoiding vector comparisons and requiring only a final dissolve step,
- **`label_stitch`** labels values as in **`label_union`**, then polygonizes each chunk independently and **stitches polygons afterward in vector space**, avoiding the need for a union–find structure,
- **`geometry_stitch`** bypasses labeling entirely by performing **polygonization on halo-expanded chunks** (1-pixel overlap), then stitches polygons similarly as in **`label_stitch`** after clipping.

GeoUtils uses **`label_stitch`** by default because it scales best in the {ref}`benchmarking-performance`.

Connectivity assumptions influence how many neighboring chunks must be used in memory, which remains small and bounded:

- **4-connectivity** (4 cardinal directions): typically 2–4 chunks,
- **8-connectivity** (adding 4 diagonals): up to 4–9 chunks.

Finally, note that **polygon stitching occurs only for polygons touching chunk boundaries**. Polygons fully contained within a chunk are produced directly without additional processing.

### Chunked rasterization

**Rasterization with chunked execution** distributes the burn operation across **output raster chunks**, allowing vector datasets to be converted into rasters without materializing the entire output array in memory.

The diagram below illustrates the execution strategy:

```{eval-rst}
.. plot:: diagrams/diagram_chunked_rasterize.py
    :width: 100%
```

For every chunk, GeoUtils first performs a **spatial query on the vector geometries**, selecting only those whose bounding boxes intersect the chunk bounds. These candidate geometries are then rasterized into the chunk-local array using {func}`~dask.array.map_blocks`. If no geometries intersect the chunk, the block function exits early and directly returns an array filled with the background value.

The memory footprint during chunked rasterization is therefore approximately limited to:

- **One output chunk array**, and
- **The subset of geometries intersecting that chunk**.

Unlike chunked reprojection or polygonization, rasterization does not require overlap or cross-chunk reconciliation, since each pixel value is determined independently of the vector intersections within the chunk.

### Chunked interpolation at points

**Interpolation at points with chunked execution** evaluates raster values at point coordinates by combining **1D chunking of the input points** with **2D chunking of the raster grid**.

The diagram below illustrates the workflow:

```{eval-rst}
.. plot:: diagrams/diagram_chunked_interp_points.py
    :width: 100%
```

First, the input points are **chunked along their 1D sequence**. For each point chunk, GeoUtils uses a fast regular-grid mapping to **raster chunks containing the corresponding point coordinates**. Only those raster chunks are processed, avoiding loading the full raster into memory.

Interpolation is then performed independently on each required raster chunk. To ensure correct interpolation near chunk boundaries, the raster chunk is **expanded by an overlap depth equal to the half-interpolation-order rounded up + 1** (for example, 1 pixels for nearest, 2 for linear and 3 for cubic).

Because points within a point chunk may fall into different raster chunks, interpolation proceeds by **looping over the intersecting raster chunks** for that point chunk. The resulting interpolated values are then **concatenated and reordered** to match the original point order.

This approach keeps memory usage low, as only the raster chunks needed for the current point chunk are loaded at any given time, and minimizes speed loss by evaluating all points belonging in a same raster chunk at once.

Finally, note that **using eager (in-memory) point coordinates is typically much faster** when the number of points is moderate. Unless necessary, keep points in memory to avoid additional chunking overhead.

### Chunked subsampling

**Chunked subsampling of valid raster values** enables selecting representative pixels from very large rasters without materializing all valid values in memory.

The diagram below illustrates the workflow:

```{eval-rst}
.. plot:: diagrams/diagram_chunked_subsample.py
    :width: 100%
```

Subsampling proceeds in **two stages**. First, GeoUtils performs a lightweight pass that counts the number of **valid pixels in each chunk**. These counts are summed to determine the final subsample size requested by the user, which may be specified either as a **fraction of valid pixels** or as an **absolute number**. This pass is inexpensive because it only requires scanning chunk masks and does not materialize the valid-value array.

Once the target sample size is known, pixels are selected using one of two strategies:
- With **`topk` (default, chunk-invariant)** sampling, each valid pixel is assigned a deterministic pseudo-random key derived from the random seed and its global linear index (row * number of columns + col). The **k smallest keys** are selected globally. Because the key depends only on the pixel index and the seed, the resulting subsample is **independent of chunk layout** and reproducible with any chunking or fully in-memory raster.
- With **`sequential` (chunk-dependent)** sampling, pixels are drawn from the **flattened sequence of valid values encountered during chunk traversal**. This strategy is typically **faster** because it avoids computing deterministic keys, but the result **depends on the chunk structure and valid-value ordering**.
