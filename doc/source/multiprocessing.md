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
(multiprocessing)=

# Multiprocessing configuration

GeoUtils implements logics from Python's [multiprocessing library](https://docs.python.org/3/library/multiprocessing.html) library
to process raster, vector and point cloud data in chunks without loading the entire dataset into memory when working with **GeoUtils objects**
({class}`~geoutils.Raster`, {class}`~geoutils.Vector` and {class}`~geoutils.PointCloud`),

When using the {class}`rst <geoutils.RasterAccessor>`, {class}`vct <geoutils.VectorAccessor>` and {class}`pc <geoutils.PointCloudAccessor>` **accessors**, use Dask instead.

See {ref}`scalability-usage` for the main Dask and Multiprocessing workflows, {ref}`scalability-concept` for the
execution model, and {ref}`scalability-support` for the operations supported by each backend.

## Using {class}`~geoutils.multiproc.MultiprocConfig`

{class}`~geoutils.multiproc.MultiprocConfig` defines chunk processing settings, such as chunks, output file, driver,
and computing cluster. Pass it as `mp_config` to a supported method to trigger out-of-memory execution.

### Raster example

```{code-cell} ipython3
import geoutils as gu
from geoutils.multiproc import MultiprocConfig

filename_rast = gu.examples.get_path("exploradores_aster_dem")
rast = gu.Raster(filename_rast)

# Reproject the raster in 200 x 200 pixel chunks
raster_config = MultiprocConfig(chunks=200)
rast_reprojected = rast.reproject(res=rast.res[0] * 2, mp_config=raster_config)

rast_reprojected
```

- **`chunks=200`**: The raster is divided into 200x200 pixel blocks.

By default, chunks are processed sequentially. The output is returned as an unloaded
{class}`~geoutils.Raster`, allowing another chunked operation to follow without loading its values.
When `outfile` is omitted, it is saved to a temporary file available through `rast_reprojected.name` until the file
is deleted or the system cleans its temporary directory.

### Adding parallel workers

To process chunks in parallel, pass a {class}`~geoutils.multiproc.MpCluster` using
{class}`~geoutils.multiproc.ClusterGenerator`:

```{code-cell} ipython3
from geoutils.multiproc import ClusterGenerator

# Use four workers and save the output to a specified file
with ClusterGenerator("multi", nb_workers=4) as cluster:
    parallel_config = MultiprocConfig(chunks=200, outfile="reprojected_parallel.tif", cluster=cluster)
    rast_reprojected_parallel = rast.reproject(res=rast.res[0] * 2, mp_config=parallel_config)

rast_reprojected_parallel
```

The `ClusterGenerator("multi", nb_workers=4)` call above creates an {class}`~geoutils.multiproc.MpCluster` internally.
By default, `MpCluster` starts workers with `forkserver` when the current platform supports it and otherwise uses
`spawn` (e.g. on Windows). To select another start method supported by the current platform, create the
cluster directly:

```{code-cell} ipython3
from geoutils.multiproc import MpCluster

# Select "spawn" method
with MpCluster({"nb_workers": 4}, start_method="spawn") as cluster:
    config = MultiprocConfig(chunks=200, cluster=cluster)
```

```{code-cell} ipython3
:tags: [remove-cell]
import os

os.remove(raster_config.outfile)
os.remove(parallel_config.outfile)
```

### Point cloud example

{meth}`~PointCloudBase.reproject` uses the same `mp_config` argument. For point cloud operations, `chunks` is an
integer number of points per task rather than raster dimensions, and the output can be GeoPackage, LAS or LAZ.

```python
points = gu.PointCloud("observations.laz")
point_config = MultiprocConfig(chunks=100_000, outfile="projected.gpkg")
projected = points.reproject(crs=32633, mp_config=point_config)

assert not points.is_loaded
assert not projected.is_loaded
```

The output format follows the filename extension, or an explicit `driver="GPKG"`, `"LAS"` or `"LAZ"`.
GeoPackage is the default when no extension is supplied. Point order and value columns are preserved; LAS/LAZ
coordinates use the file's stored precision. The accessor interfaces use their normal eager or lazy Dask path without
`mp_config`.

## Advanced block functions

GeoUtils offers Dask-named functions for custom out-of-memory multiprocessing. The naming mirrors Dask arrays and
dataframes, but multiprocessing results are computed eagerly and cannot remain lazy.

### {func}`~geoutils.multiproc.map_overlap`: process and save large rasters

This function applies a user-defined function to raster blocks and **saves the output** to a file. The entire raster is
**never loaded into memory at once**. It returns the raster metadata loaded from the file.

```{code-cell} ipython3
import numpy as np
import scipy
from geoutils.multiproc import map_overlap

config_basic = MultiprocConfig(chunks=200, outfile="output.tif")

def filter(raster: gu.Raster, size: int) -> gu.Raster:
    new_data = scipy.ndimage.maximum_filter(raster.data, size)
    if raster.nodata is not None:
        new_data = np.ma.masked_equal(new_data, raster.nodata)
    raster.data = new_data
    return raster

size = 1
raster_filtered = map_overlap(filter, filename_rast, config_basic, size, depth=size+1)
raster_filtered
```

### {func}`~geoutils.multiproc.map_blocks`: extract and collect data from large rasters

This function applies a function to raster blocks and **returns a list** of extracted data without saving a new raster
file. It is useful for summary statistics, features, or other results that do not return a raster.

```{code-cell} ipython3
from typing import Any
from geoutils.multiproc import map_blocks

# Compute mean
def compute_statistics(raster: gu.Raster) -> dict[str, np.floating[Any]]:
    return raster.stats(["mean", "valid_count"])

stats_results = map_blocks(compute_statistics, filename_rast, config_basic)
total_count = sum([stats["valid_count"] for stats in stats_results])
total_mean = sum([stats["mean"] * stats["valid_count"] for stats in stats_results]) / total_count
print("Mean: ", total_mean)
```

```{code-cell} ipython3
:tags: [remove-cell]
os.remove(config_basic.outfile)
```

```{note}
To include block location in the results, set `return_block_info=True`.
```

### Choosing the right function

| Use case                                      | Function                                                                     |
|-----------------------------------------------|------------------------------------------------------------------------------|
| Apply processing and save results as a raster | {func}`~geoutils.multiproc.map_overlap`                                      |
| Extract statistics or features into a list    | {func}`~geoutils.multiproc.map_blocks`                                       |
| Track block locations with extracted data     | {func}`~geoutils.multiproc.map_blocks` with `return_block_info=True`         |
