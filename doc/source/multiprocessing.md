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

# Multiprocessing

## Overview

Processing large raster datasets can be **computationally expensive and memory-intensive**. To optimize performance and enable **out-of-memory processing**, GeoUtils provides **multiprocessing utilities** that allow users to process raster data in parallel by splitting it into blocks.

GeoUtils offers Dask-named functions for out-of-memory multiprocessing. The naming mirrors Dask arrays/dataframes for easier exchange and documentation, but multiprocessing results are computed eagerly and cannot remain lazy.

- {func}`~geoutils.multiproc.map_overlap`: Applies a function to raster blocks and **saves the output** as a {class}`geoutils.Raster`.
- {func}`~geoutils.multiproc.map_blocks`: Applies a function and **collects extracted data** from raster blocks into a list.

Both functions require a **multiprocessing configuration** defined with {class}`~geoutils.raster.MultiprocConfig`.

---

## Using {class}`~geoutils.multiproc.MultiprocConfig`

{class}`~geoutils.multiproc.MultiprocConfig` defines block processing settings, such as chunks, output file, driver, and computing cluster. It ensures that computations are performed **without loading the entire raster into memory**.

### Example: creating a {class}`~geoutils.multiproc.MultiprocConfig` object
```{code-cell} ipython3
from geoutils.multiproc import ClusterGenerator
from geoutils.multiproc import MultiprocConfig

# Create a configuration without multiprocessing cluster (tasks will be processed sequentially).
config_basic = MultiprocConfig(chunks=200, outfile="output.tif", cluster=None)

# Create a configuration with a multiprocessing cluster
config_np = config_basic.copy()
config_np.cluster = ClusterGenerator("multi", nb_workers=4)
```
- **`chunks=200`**: The raster is divided into 200x200 pixel blocks.
- **`outfile="output.tif"`**: The results will be saved under this file (if not provided, temporary file by default).
- **`cluster=ClusterGenerator("multi", nb_workers=4)`**: Enables parallel processing.

---

## {func}`~geoutils.multiproc.map_overlap`: process and save large rasters

This function applies a user-defined function to raster blocks and **saves the output** to a file. The entire raster is **never loaded into memory at once**, making it suitable for processing large datasets.
The function returned the raster metadata loaded from the file.

### When to use
- When the function **returns a Raster**.
- When the result should be **saved as a new raster**.
- When working with large rasters that do not fit into memory.

### Example: applying a raster filter
```{code-cell} ipython3
import geoutils as gu
import scipy
import numpy as np
from geoutils.multiproc import map_overlap

filename_rast = gu.examples.get_path("exploradores_aster_dem")

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

```{code-cell} ipython3
:tags: [remove-cell]
import os
os.remove(config_basic.outfile)
```

---

## {func}`~geoutils.multiproc.map_blocks`: extract and collect data from large rasters

This function applies a function to raster blocks and **returns a list** of extracted data, without saving a new raster file. The process runs in **out-of-memory mode**, ensuring efficient handling of large datasets.

### When to use
- When the function **does not return a Raster**.
- When extracting **summary statistics, features, or analysis results**.
- When processing large rasters that cannot fit into memory.

### Example: extracting elevation statistics
```{code-cell} ipython3
from geoutils.multiproc import map_blocks
from typing import Any

# Compute mean

def compute_statistics(raster: gu.Raster) -> dict[str, np.floating[Any]]:
    return raster.get_stats(stats_name=["mean", "valid_count"])

stats_results = map_blocks(compute_statistics, filename_rast, config_basic)
total_count = sum([stats["valid_count"] for stats in stats_results])
total_mean = sum([stats["mean"] * stats["valid_count"] for stats in stats_results]) / total_count
print("Mean: ", total_mean)
```

```{Note}
To include block location in the results, set `return_block_info=True`.
```

---

## Choosing the right function

| Use case                                      | Function                                                                     |
|-----------------------------------------------|------------------------------------------------------------------------------|
| Apply processing and save results as a raster | {func}`~geoutils.multiproc.map_overlap`                                      |
| Extract statistics or features into a list    | {func}`~geoutils.multiproc.map_blocks`                                       |
| Track block locations with extracted data     | {func}`~geoutils.multiproc.map_blocks` with `return_block_info=True`         |
