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
(scalability-usage)=
# Usage and good practices

GeoUtils supports scalable execution for most of its **raster** and (soon) **point cloud** operations (**vector** support may be added in the future, but is usually less limiting).

It relies on two execution backends:

- **Dask**, through its {class}`rst <geoutils.RasterAccessor>` Xarray accessor and `pc` Pandas accessor (**lazy** and **chunked** execution),
- **Multiprocessing**, through its {class}`~geoutils.Raster` and {class}`~geoutils.PointCloud` objects (**chunked** execution only) .

Both backends mirror the **exact same object operations and chunked logic**, and yield **identical** results as in-memory operations.
**Lazy** refers to deferred execution using Dask, while **chunked** refers to processing raster data tile-by-tile to limit memory usage.
For details on scalability concepts, see the {ref}`scalability-concept` page.

As a rule of thumb:

- Use **Dask** to work on **Xarray and GeoPandas objects** through our accessors {class}`rst <geoutils.RasterAccessor>` and `pc`, and if you want to chain several operations lazily.
- Use **Multiprocessing** to work with our {class}`~geoutils.Raster` and {class}`~geoutils.PointCloud` objects, and if you are fine with intermediate writing/reading between steps.
- Use standard **in-memory execution** to work efficiently on small rasters, which is possible even if those were loaded from larger rasters (use {class}`~geoutils.Raster.crop`).

```{note}
GeoUtils currently targets **scalable CPU execution**. However, as many of our numerical operations rely on **NumPy**, **SciPy** or **Numba**, those are planned to be linked to their **GPU** counterparts (**CuPy** and **Numba CUDA**).
```

## Using Dask through accessors

With **Dask**, raster operations are both **chunked** and **lazy**.
This behavior is enabled by opening a raster with the `chunks` argument, which returns an Xarray object backed by Dask arrays.

```{code-cell} python
import geoutils as gu

filename_rast = gu.examples.get_path("exploradores_aster_dem")

ds = gu.open_raster(filename_rast, chunks={"x": 200, "y": 200})

ds
```

GeoUtils, through the {class}`rst <geoutils.RasterAccessor>` accessor, automatically detects the **Dask** input and switches to a chunked implementation.

```{code-cell} python
# Change output resolution
out_res = (ds.rst.res[0] * 2, ds.rst.res[1] / 2)

# Reproject lazily and out-of-memory
ds_reproj = ds.rst.reproject(
    res=out_res,
    resampling="bilinear",
)

ds_reproj
```

The resulting raster remains **lazy**. Computation only happens when explicitly requested with {meth}`~dask.array.Array.compute()`.

For a raster output, one typically wants to write to file lazily to avoid loading it in-memory:

```{code-cell} python
# ds_reproj.rst.to_file("reproj_rast.tif", compute=True)
```

This triggers computation and writes the raster **chunk-by-chunk** to the output file.

Or, the output can be chained with another operation. For example, we can extract a small random subsample of the reprojected raster:

```{code-cell} python
# Subsample lazily and out-of-memory
sub_ds = ds_reproj.rst.subsample(
    subsample=5000,
)

sub_ds
```

The output array is again lazy, and in this case we can use {meth}`~dask.array.Array.compute()` to return the in-memory NumPy array:
```{code-cell} python
sub_ds.compute()
```

Lazy execution is particularly useful to **chain several operations** without ever loading the full raster into memory or writing any file to disk.

## Using Multiprocessing through GeoUtils objects

Chunked execution can also be enabled on the {class}`~geoutils.Raster` object using Multiprocessing.
Multiprocessing performs **chunked execution** (only loads chunks in-memory), but is not **lazy** (it runs immediately).

```{code-cell} python
rast = gu.Raster(filename_rast)

rast
```

By passing a {class}`~geoutils.multiproc.MultiprocConfig` configuration to an operation, the out-of-memory behaviour is triggered.
The configuration requires a `chunk_size`, and can optionally define the output file and cluster to use (defaults to temporary instances for both).

```{code-cell} python
from geoutils.multiproc import MultiprocConfig

# Optional: specify output file (otherwise a temporary file is created)
mp_config = MultiprocConfig(chunk_size=200, outfile="reproj_rast.tif")

# Reproject out-of-memory, reading and writing chunk-by-chunk
rast_reproj_mp = rast.reproject(
    res=out_res,
    resampling="bilinear",
    mp_config=mp_config,
)

rast_reproj_mp
```

If the output is a {class}`~geoutils.Raster`, it is written to disk out-of-memory, and the returned object is a {class}`~geoutils.Raster` of that file without data loaded.
This keeps syntax consistent with in-memory code, and allow to easily chain operations. 

For other output types, the Multiprocessing backends will load the result in-memory. 

```{code-cell} python
# Subsample out-of-memory and return loaded array
samp_rast_mp = rast_reproj_mp.subsample(
    subsample=5000,
)

samp_rast_mp
```

```{code-cell} ipython3
:tags: [remove-cell]
import os
os.remove(mp_config.outfile)
```

This backend is convenient when working directly with {class}`~geoutils.Raster` objects and performing **step-by-step processing**.

## Good practices with chunked and lazy operations

- If **memory** is the limitating factor for you, use a **single-threaded scheduler** through Dask (```dask.config.set(scheduler='single-threaded')```) or Multiprocessing (default cluster),
- If **speed** is the limiting factor for you, use **parallelized processes** through Dask (see [Dask scheduler configuration](https://docs.dask.org/en/stable/scheduler-overview.html#scheduler-overview)) or Multiprocessing (see our Cluster configuration),
- Choose chunk sizes large enough to reduce scheduling overhead, but **small enough to fit comfortably in memory**,
- Check that your data files have **on-disk chunksizes** (otherwise loads everything) and use a multiple of it for optimal **in-memory chunking**,
- Keep chunk sizes **consistent across operations** to avoid unnecessary rechunking,
- Insert **breakpoints** (for example by writing intermediate results to disk) to prevent building overly large Dask graphs.

For more guidance on chunk sizing and performance, see the [Dask array best practices](https://docs.dask.org/en/stable/array-best-practices.html).

Finally, note that currently, operations returning **point** or **vector** outputs are often **eager** and scalable execution applies mostly to the **raster input/output**.
The full description of supported methods is available on the {ref}`scalability-support` page.