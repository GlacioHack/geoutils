(scalability-concept)=
# Concept definitions

This section describes scalability concepts important to grasp to manipulate our objects.

Scalable execution relies on three complementary mechanisms:

- **Deferred I/O with implicit loading:** Operations that update data-related metadata without loading the underlying array or geometries,
- **Chunked execution:** Operations that process data tile-by-tile to limit memory usage,
- **Lazy execution:** Operations whose computation is deferred until explicitly requested.

These mechanisms are often combined (e.g., Dask operations are always chunked **and** lazy) but are conceptually independent.

Finally, one should note that the above concepts only apply to operations that interact with the underlying **data arrays or geometries** of GeoUtils objects.
Naturally, all **metadata operations** (e.g., accessing {attr}`~geoutils.Raster.crs`, {attr}`~geoutils.Raster.bounds`, or {meth}`~geoutils.Raster.info`) have no effect on the array, and do not trigger any loading.

## Deferred I/O and implicit loading

**Deferred input/output** refers to operations that modify only **internal I/O metadata**, avoid reading the data entirely and postponing loading.

Typical examples include {meth}`~geoutils.Raster.crop`, {meth}`~geoutils.Raster.copy`, and {meth}`~geoutils.Raster.translate`, 
which behave similarly as Xarray's {meth}`~xarray.DataArray.sel`, {meth}`~xarray.DataArray.copy`, or {meth}`~xarray.DataArray.assign_coords`.

When using the Xarray `rst` accessor, this behavior follows the **native Xarray deferred I/O model**. The {class}`~geoutils.Raster` class implements the
same behavior so that both APIs have consistent semantics.

This behaviour pairs intrinsically with **implicit loading:** When an object is opened, only metadata is loaded. 
Accessing {attr}`~geoutils.Raster.data`, or calling operations that require the array, will **implicitly load the data into memory**.

An important aspect of **deferred I/O** is that it works with both **in-memory** (NumPy) and **scalable backends** (Dask), allowing 
to extract parts of large files without any chunked or lazy considerations.

## Chunked execution

**Chunked execution** refers to processing raster data **tile-by-tile** instead of loading the full array into memory.

This enables **out-of-core execution**, allowing datasets larger than available RAM to be processed safely.

In GeoUtils, chunked execution is implemented through two backends:

- **Dask**, used through the Xarray `rst` accessor,
- **Multiprocessing**, used through the {class}`~geoutils.Raster` object.

Both backends read and process raster chunks sequentially, keeping peak memory usage proportional to the chunk size rather than the full dataset size.
Chunked execution therefore allows GeoUtils to scale to large datasets while maintaining a **predictable memory footprint**. 
For a list of expected memory usage per operation, see the {ref}`scalability-support` page.

## Lazy execution

Lazy execution refers to **deferring computation until results are explicitly requested**.

In GeoUtils, lazy execution is available through the Xarray `rst` accessor with **Dask-backed arrays**.

Operations build a **Dask computation graph** instead of executing immediately. The computation is triggered only when required, for example when calling
`compute()` or when writing results to disk. It is particularly useful when **chaining multiple raster operations**, because intermediate results do not need to be materialized or 
written/read from disk (which costs extra I/O time, often much longer than compute time).

Lazy execution always relies on **chunked execution**, but the reverse is not true: chunked processing can also run eagerly, as in the Multiprocessing backend.