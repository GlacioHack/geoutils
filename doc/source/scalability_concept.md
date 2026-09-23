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
(scalability-concept)=
# Concept definitions

This section describes scalability concepts that are important to grasp to efficiently manipulate geospatial objects.

Scalable execution relies on three complementary mechanisms:

- {ref}`Deferred I/O with implicit loading <core-lazy-load>`: Operations that update data-related metadata without loading the underlying array or geometries,
- **Chunked execution:** Operations that process raster tiles or dataframe partitions separately to limit memory usage,
- **Lazy execution:** Operations whose computation is deferred until explicitly requested.

These mechanisms are often combined (e.g., Dask operations are always **chunked and lazy**) but are conceptually independent.

```{note}
The above concepts only apply to operations that interact with the underlying **data arrays or geometries** of objects.
Naturally, all **metadata operations** (e.g., accessing {attr}`~RasterBase.crs`, {attr}`~RasterBase.bbox`, or {meth}`~RasterBase.info`) have no effect on the array, and therefore do not trigger any loading or scalable execution.
```

## Chunked execution

**Chunked execution** refers to processing raster data **tile-by-tile** or vector and point data **partition-by-partition** instead of loading the full dataset into memory.

This enables **out-of-core execution**, allowing datasets larger than available RAM to be processed safely.

In GeoUtils, chunked execution is implemented through two backends:

- **Dask**, used through the Xarray {class}`rst <geoutils.RasterAccessor>` and GeoPandas {class}`vct <geoutils.VectorAccessor>` and {class}`pc <geoutils.PointCloudAccessor>` accessors,
- **Multiprocessing**, used through supported methods of the {class}`~geoutils.Raster`, {class}`~geoutils.Vector` and {class}`~geoutils.PointCloud` objects.

Both backends read and process chunks separately, keeping peak memory usage proportional to the chunk size rather than the full dataset size.
Chunked execution therefore allows GeoUtils to scale to large datasets while maintaining a **predictable memory footprint**.
For a list of expected memory usage per operation, see the {ref}`scalability-support` page.

```{code-cell}
import geoutils as gu

# Get the example raster filename
filename_rast = gu.examples.get_path("exploradores_aster_dem")

# Open raster (data is not loaded)
rast = gu.Raster(filename_rast)

# Create Multiprocessing config, output filepath optional (temporary file by default)
mp_config = gu.multiproc.MultiprocConfig(chunks=200)

# Filter raster with a gaussian in a chunked manner through Multiprocessing
rast_filt = rast.filter(method="gaussian", sigma=4, mp_config=mp_config)

# The operation happened out-of-memory in chunk-by-chunk
print(f"Temporary raster file created during operation: {rast_filt.name}")
print(f"Is input raster loaded after filtering? {rast.is_loaded}")
print(f"Is output raster loaded after filtering? {rast_filt.is_loaded}")
```

## Lazy execution

Lazy execution refers to **deferring computation until results are explicitly requested**.

In GeoUtils, lazy execution is available through the Xarray {class}`rst <geoutils.RasterAccessor>` accessor with
**Dask arrays**, and the GeoPandas {class}`vct <geoutils.VectorAccessor>` and
{class}`pc <geoutils.PointCloudAccessor>` accessors with **Dask dataframes**.

Operations build a **Dask computation graph** instead of executing immediately. The computation is triggered only when required, for example when calling
`compute()` directly or when writing results to disk. It is particularly useful when **chaining multiple operations**, because intermediate results do not need to be materialized or
written/read from disk (which costs extra I/O time, often much longer than compute time).

Lazy execution always relies on **chunked execution**, but the reverse is not true: chunked processing can also run eagerly, as in the Multiprocessing backend.

```{code-cell}
# Open raster lazily with chunks (automatically enables Dask)
ds = gu.open_raster(filename_rast, chunks={"x": 200, "y": 200})

print("Input is lazy (Dask arrays):\n")
ds
```

If the output returned is still a Dask array, the operation was lazy.

```{code-cell}
# Interpolate 30 points from array in chunk-by-chunk
import numpy as np
rng = np.random.default_rng(seed=42)
x = rng.uniform(ds.rst.bbox.left, ds.rst.bbox.right, size=30)
y = rng.uniform(ds.rst.bbox.bottom, ds.rst.bbox.top, size=30)
ds_interp = ds.rst.interp_points((x, y), as_array=True)

# Result is still lazy
print("Result is still lazy after raster interpolation:\n")
ds_interp
```

We can materialize it with `compute()`, which now returns a NumPy array:

```{code-cell}
ds_interp = ds_interp.compute()
ds_interp
```
