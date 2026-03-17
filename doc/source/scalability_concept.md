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

- **Deferred I/O with implicit loading:** Operations that update data-related metadata without loading the underlying array or geometries,
- **Chunked execution:** Operations that process data tile-by-tile to limit memory usage,
- **Lazy execution:** Operations whose computation is deferred until explicitly requested.

These mechanisms are often combined (e.g., Dask operations are always **chunked and lazy**) but are conceptually independent.

```{note}
The above concepts only apply to operations that interact with the underlying **data arrays or geometries** of objects.
Naturally, all **metadata operations** (e.g., accessing {attr}`~geoutils.Raster.crs`, {attr}`~geoutils.Raster.bounds`, or {meth}`~geoutils.Raster.info`) have no effect on the array, and therefore do not trigger any loading or scalable execution.
```

## Deferred I/O and implicit loading

**Deferred input/output** refers to operations that modify only **internal I/O metadata**, avoiding reading the data entirely and postponing loading.

Typical examples include {meth}`~geoutils.Raster.crop`, {meth}`~geoutils.Raster.copy`, and {meth}`~geoutils.Raster.translate`, 
which behave similarly as Xarray's {meth}`~xarray.DataArray.sel`, {meth}`~xarray.DataArray.copy`, or {meth}`~xarray.DataArray.assign_coords`.

When using the Xarray {class}`rst <geoutils.RasterAccessor>` accessor, this behavior follows the **native Xarray deferred I/O model**. The {class}`~geoutils.Raster` class implements the
same behavior so that both APIs have consistent semantics. 

An important aspect of **deferred I/O** is that it works with both **in-memory** (NumPy) and **scalable backends** (Dask), allowing 
to extract parts of large files without any chunked or lazy considerations.

```{code-cell}
import geoutils as gu

# We open the dataset without Dask backend (same behaviour with Raster class)
filename_rast = gu.examples.get_path("exploradores_aster_dem")
ds = gu.open_raster(filename_rast)

# We crop the data
ds_cropped = ds.rst.icrop((0, 0, 100, 100))

# Neither input nor output dataset are loaded yet
print(f"Input loaded by deferred I/O? {ds.rst.is_loaded}")
print(f"Output loaded by deferred I/O? {ds_cropped.rst.is_loaded}")
```

This behaviour pairs intrinsically with **implicit loading:** When an object is opened, only metadata is loaded. 
Accessing {attr}`~geoutils.Raster.data`, or calling operations that require the underlying array or geometries will **implicitly load the data into memory**.

```{code-cell}
# Is the above dataset loaded?
print(f"Loaded before data operation? {ds.rst.is_loaded}")

# We compute statistics, which loads the array
ds.rst.get_stats()

# The dataset is now loaded
print(f"Loaded after data operation? {ds.rst.is_loaded}")
```

## Chunked execution

**Chunked execution** refers to processing raster data **tile-by-tile** instead of loading the full array into memory.

This enables **out-of-core execution**, allowing datasets larger than available RAM to be processed safely.

In GeoUtils, chunked execution is implemented through two backends:

- **Dask**, used through the Xarray {class}`rst <geoutils.RasterAccessor>` accessor,
- **Multiprocessing**, used through the {class}`~geoutils.Raster` object.

Both backends read and process raster chunks sequentially, keeping peak memory usage proportional to the chunk size rather than the full dataset size.
Chunked execution therefore allows GeoUtils to scale to large datasets while maintaining a **predictable memory footprint**. 
For a list of expected memory usage per operation, see the {ref}`scalability-support` page.

```{code-cell}
# Open raster (data is not loaded)
rast = gu.Raster(filename_rast)

# Create Multiprocessing config, output filepath optional (temporary file by default)
mp_config = gu.multiproc.MultiprocConfig(chunk_size=200)

# Filter raster with a gaussian in a chunked manner through Multiprocessing
rast_filt = rast.filter(method="gaussian", sigma=4, mp_config=mp_config)

# The operation happened out-of-memory in chunk-by-chunk
print(f"Temporary raster file created during operation: {rast_filt.name}")
print(f"Is input raster loaded? {rast.is_loaded}")
print(f"Is output raster loaded? {rast_filt.is_loaded}")
```

## Lazy execution

Lazy execution refers to **deferring computation until results are explicitly requested**.

In GeoUtils, lazy execution is available through the Xarray {class}`rst <geoutils.RasterAccessor>` accessor with **Dask-backed arrays**.

Operations build a **Dask computation graph** instead of executing immediately. The computation is triggered only when required, for example when calling
`compute()` or when writing results to disk. It is particularly useful when **chaining multiple raster operations**, because intermediate results do not need to be materialized or 
written/read from disk (which costs extra I/O time, often much longer than compute time).

Lazy execution always relies on **chunked execution**, but the reverse is not true: chunked processing can also run eagerly, as in the Multiprocessing backend.

```{code-cell}
# Open raster lazily with chunks (enables Dask)
ds = gu.open_raster(filename_rast, chunks={"x": 200, "y": 200})

print("Input is lazy (Dask arrays):\n")
ds
```

```{code-cell}
# Interpolate 30 points from array in chunk-by-chunk
import numpy as np
rng = np.random.default_rng(seed=42)
x = rng.uniform(ds.rst.bounds.left, ds.rst.bounds.right, size=30)
y = rng.uniform(ds.rst.bounds.bottom, ds.rst.bounds.top, size=30)
ds_interp = ds.rst.interp_points((x, y), as_array=True)

# Result is still lazy
print("Result is still lazy after raster interpolation:\n")
ds_interp
```

We can materialize it with `compute()`:

```{code-cell}
ds_interp.compute()
```