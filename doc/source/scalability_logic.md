(scalability-logic)=
# Implementation strategies

Implementing **chunked execution** requires developping substantial internal logic often invisible to the user, making it difficult to understand what is happening in the background and how to potentially address a scalability issue.

Additionally, for certain methods, there is often no single best solution. We therefore propose several **strategies** to further improve performance based on the nature of the input data.

Below, we detail the logic behind our **chunked execution** implementations, both as an educational resource and to help optimize your code (such as memory usage).

## Summary

Several operations are easy to support for **chunked execution** as they directly re-use existing **Dask** methods:
- The {meth}`~geoutils.Raster.filter` function uses {func}`~dask.array.map_overlap` with a `depth` (overlap) half the `size` of the filter,
- The {meth}`~geoutils.Raster.proximity` function uses {func}`~dask.array.map_overlap` with a `max_distance` parameter.

Other operations are more complex and require specific logic {ref}`specific logic described further below<specific-logic>` and summarized as:
- The {meth}`~geoutils.Raster.reproject` function **maps the intersection of projected source grid chunks for each destination chunk** (with potentially different CRS, resolution and bounds), and defines default output chunksizes based on resolution change to avoid unexpected memory blowup,
- The {meth}`~geoutils.Raster.polygonize` function polygonizes implements **three chunk-boundary reconciliation strategies** with different considerations for connected-component labeling and stitching of geometries,
- The {meth}`~geoutils.Vector.rasterize` function performs a geometry subsetting then directly **maps rasterized output blocks** only utilizing these subset geometries.
- The {meth}`~geoutils.Raster.interp_points` function performs a **fast regular-grid mapping of point locations in raster chunks**, expanding raster chunks by a few pixels depending on the resampling method, then performing ordered concatenation of outputs,
- The {meth}`~geoutils.Raster.subsample` function performs an initial chunk-by-chunk sum of valid values to define the requested sample size, then samples values per chunk through **reproducible chunk-invariant seeding** (default) or **faster chunk-dependent seeding**.

(specific-logic)=
## Logic of implementations

### Chunked reprojection

**Reprojection with chunked execution** requires mapping **destination chunks**—defined on a regular grid in the output CRS and potentially with different resolution or bounds—to **projected source chunks**, which become geometrically deformed after reprojection.

The diagram below illustrates this mapping procedure, with a new CRS and a downsampling of 2:

```{eval-rst}
.. plot:: code/diagram_chunked_reproject.py
    :width: 100%
```


During reprojection, destination chunks are **expanded by 3 pixels** to ensure adequate resampling at chunk boundaries. As a result of this and deformations, the number of source chunks used at once in memory is therefore **always 4-9**.

Unlike most chunked operations, **reprojection does not preserve identical input and output chunk sizes**. When the resolution changes, maintaining the same chunk size would cause the number of intersecting source chunks to scale approximately with **the square of the downsampling factor** (because reprojection operates in two spatial dimensions). For example, a downsampling factor of 2 would cause four times more source chunks to be accessed per destination chunk, which would quickly increase memory usage during coarse reprojection.

To prevent this, GeoUtils automatically **scales the output chunk size according to the resolution change**, keeping the number of source chunks involved in each operation bounded.

Finally, note that **chunked reprojection with GCPs or RCPs is currently not supported**.

### Chunked polygonization

**Polygonization with chunked execution** requires identifying raster regions that may extend across **chunk boundaries**. Because chunks are processed independently, connected regions intersecting a boundary must later be **reconciled across neighboring chunks** to produce correct polygons.

The diagram below illustrates the three strategies implemented in GeoUtils for performing this reconciliation:

```{eval-rst}
.. plot:: code/diagram_chunked_polygonize.py
    :width: 100%
```

All strategies begin by processing individual raster chunks independently, then reconstruct continuous polygons that span chunk boundaries.

Conceptually, the methods differ in how cross-chunk regions are reconstructed: 
- **`label_union`** labels values in each chunk, then finds the **union of matching labels across chunk seams** before polygonization, avoiding vector comparisons and requiring only a final dissolve step,
- **`label_stitch`** labels values as in **`label_union`**, then polygonizes each chunk independently and **stitches polygons afterward in vector space**, avoiding the need for a union–find structure,
- **`geometry_stitch`** bypasses labeling entirely by performing **polygonization on halo-expanded chunks** (1-pixel overlap), then stitches polygons similarly as in **`label_stitch`** after clipping.

Connectivity assumptions influence how many neighboring chunks must be used in memory, which remains small and bounded:

- **4-connectivity** (4 cardinal directions): typically 2–4 chunks,
- **8-connectivity** (adding 4 diagonals): up to 4–9 chunks.

Finally, note that **polygon stitching occurs only for polygons touching chunk boundaries**. Polygons fully contained within a chunk are produced directly without additional processing.

### Chunked rasterization

**Rasterization with chunked execution** distributes the burn operation across **output raster chunks**, allowing vector datasets to be converted into rasters without materializing the entire output array in memory.

The diagram below illustrates the execution strategy:

```{eval-rst}
.. plot:: code/diagram_chunked_rasterize.py
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
.. plot:: code/diagram_chunked_interp_points.py
    :width: 100%
```

First, the input points are **chunked along their 1D sequence**. For each point chunk, GeoUtils uses a fast regular-grid mapping to **raster chunks containing the corresponding point coordinates**. Only those raster chunks are processed, avoiding loading the full raster into memory.

Interpolation is then performed independently on each required raster chunk. To ensure correct interpolation near chunk boundaries, the raster chunk is **expanded by an overlap depth equal to the half-interpolation-order rounded up + 1** (for example, 1 pixels for nearest, 2 for linear and 3 for cubic).

Because points within a point chunk may fall into different raster chunks, interpolation proceeds by **looping over the intersecting raster chunks** for that point chunk. The resulting interpolated values are then **concatenated and reordered** to match the original point order.

This approach keeps memory usage low, as only the raster chunks needed for the current point chunk are loaded at any given time.

Finally, note that **using eager (in-memory) point coordinates is typically much faster** when the number of points is moderate. Unless necessary, keep points in memory to avoid additional chunking overhead.

### Chunked subsampling

**Chunked subsampling of valid raster values** enables selecting representative pixels from very large rasters without materializing all valid values in memory.

The diagram below illustrates the workflow:

```{eval-rst}
.. plot:: code/diagram_chunked_subsample.py
    :width: 100%
```

Subsampling proceeds in **two stages**. First, GeoUtils performs a lightweight pass that counts the number of **valid pixels in each chunk**. These counts are summed to determine the final subsample size requested by the user, which may be specified either as a **fraction of valid pixels** or as an **absolute number**. This pass is inexpensive because it only requires scanning chunk masks and does not materialize the valid-value array.

Once the target sample size is known, pixels are selected using one of two strategies:
- With **`topk` (chunk-invariant)** sampling, each valid pixel is assigned a deterministic pseudo-random key derived from the random seed and its global linear index (row + col * number of rows). The **k smallest keys** are selected globally. Because the key depends only on the pixel index and the seed, the resulting subsample is **independent of chunk layout** and reproducible with any chunking or fully in-memory raster.
- With **`sequential` (chunk-dependent)** sampling, pixels are drawn from the **flattened sequence of valid values encountered during chunk traversal**. This strategy is typically **faster** because it avoids computing deterministic keys, but the result **depends on the chunk structure and valid-value ordering**.
