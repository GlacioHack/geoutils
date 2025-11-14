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

Processing large raster datasets can be **computationally expensive and memory-intensive**. To optimize performance and enable **out-of-memory processing**, GeoUtils provides **multiprocessing utilities** that allow users to process raster data in parallel by splitting it into tiles.

GeoUtils offers two functions for out-of-memory multiprocessing:

- {func}`~geoutils.raster.map_overlap_multiproc_save`: Applies a function to raster tiles and **saves the output** as a {class}`geoutils.Raster`.
- {func}`~geoutils.raster.map_multiproc_collect`: Applies a function and **collects extracted data** from raster tiles into a list.

Both functions require a **multiprocessing configuration** defined with {class}`~geoutils.raster.MultiprocConfig`.

---

## Using {class}`~geoutils.raster.MultiprocConfig`

{class}`~geoutils.raster.MultiprocConfig` defines tiling and processing settings, such as chunk size, output file, driver, and computing cluster. It ensures that computations are performed **without loading the entire raster into memory**.

### Example: creating a {class}`~geoutils.raster.MultiprocConfig` object
```{code-cell} ipython3
from geoutils.raster import ClusterGenerator
from geoutils.raster import MultiprocConfig

# Create a configuration without multiprocessing cluster (tasks will be processed sequentially)
config_basic = MultiprocConfig(chunk_size=200, outfile="output.tif", cluster=None)

# Create a configuration with a multiprocessing cluster
config_np = config_basic.copy()
config_np.cluster = ClusterGenerator("multi", nb_workers=4)
```
- **`chunk_size=200`**: The raster is divided into 200x200 pixel tiles.
- **`outfile="output.tif"`**: The results will be saved under this file (if not provided, temporary file by default).
- **`cluster=ClusterGenerator("multi", nb_workers=4)`**: Enables parallel processing.

---

## {func}`~geoutils.raster.map_overlap_multiproc_save`: process and save large rasters

This function applies a user-defined function to raster tiles and **saves the output** to a file. The entire raster is **never loaded into memory at once**, making it suitable for processing large datasets.
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
from geoutils.raster import RasterType, map_overlap_multiproc_save

filename_rast = gu.examples.get_path("exploradores_aster_dem")

def filter(raster: RasterType, size: int) -> RasterType:
    new_data = scipy.ndimage.maximum_filter(raster.data, size)
    if raster.nodata is not None:
        new_data = np.ma.masked_equal(new_data, raster.nodata)
    raster.data = new_data
    return raster

size = 1
raster_filtered = map_overlap_multiproc_save(filter, filename_rast, config_basic, size, depth=size+1)
raster_filtered
```

```{code-cell} ipython3
:tags: [remove-cell]
import os
os.remove(config_basic.outfile)
```

---

## {func}`~geoutils.raster.map_multiproc_collect`: extract and collect data from large rasters

This function applies a function to raster tiles and **returns a list** of extracted data, without saving a new raster file. The process runs in **out-of-memory mode**, ensuring efficient handling of large datasets.

### When to use
- When the function **does not return a Raster**.
- When extracting **summary statistics, features, or analysis results**.
- When processing large rasters that cannot fit into memory.

### Example: extracting elevation statistics
```{code-cell} ipython3
from geoutils.raster import map_multiproc_collect
from typing import Any

# Compute mean

def compute_statistics(raster: gu.Raster) -> dict[str, np.floating[Any]]:
    return raster.get_stats(stats_name=["mean", "valid_count"])

stats_results = map_multiproc_collect(compute_statistics, filename_rast, config_basic)
total_count = sum([stats["valid_count"] for stats in stats_results])
total_mean = sum([stats["mean"] * stats["valid_count"] for stats in stats_results]) / total_count
print("Mean: ", total_mean)
```

```{Note}
To include tile location (col_min, col_max, row_min, row_max) in the results, set `return_tile=True`.
```

---

## Choosing the right function

| Use case                                      | Function                                                               |
|-----------------------------------------------|------------------------------------------------------------------------|
| Apply processing and save results as a raster | {func}`~geoutils.raster.map_overlap_multiproc_save`                 |
| Extract statistics or features into a list    | {func}`~geoutils.raster.map_multiproc_collect`                         |
| Track tile locations with extracted data      | {func}`~geoutils.raster.map_multiproc_collect` with `return_tile=True` |
