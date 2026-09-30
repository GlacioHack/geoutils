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
(filters)=

# Filters
GeoUtils provides filters for {class}`~geoutils.Raster` and {class}`~geoutils.PointCloud` data.

## Available filters
The following filters are currently available in GeoUtils:

| Filter Name | Description                                                                                                                                                           | Typical Effect                                                  |
|:------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------|
| `gaussian`  | Applies a Gaussian (blur) filter with a specified sigma.                                                                                                              | Smooths the image, reduces noise while slightly blurring edges. |
| `median`    | Applies a median filter over a sliding window.                                                                                                                        | Reduces noise while preserving edges better than Gaussian.      |
| `mean`      | Applies a mean (average) filter with a specified kernel size.                                                                                                         | Smooths the image uniformly, reduces high-frequency noise.      |
| `max`       | Applies a maximum filter over a sliding window.                                                                                                                       | Enhances bright regions, expands high-intensity areas.          |
| `min`       | Applies a minimum filter over a sliding window.                                                                                                                       | Suppresses bright regions, expands dark regions. |
| `distance`  | Removes pixels that deviate strongly from local neighborhood average (within a radius).                                                                               | Removes outliers and anomalous values based on local context.   |
| `custom`    | Allows users to define their own filter function to be applied to a numpy array.                                                                                      |                                                                 |

## Parameters
| Parameter           | Definition                                                                                 | Available for filter           | Type  | Default value |
|:--------------------|--------------------------------------------------------------------------------------------|--------------------------------|-------|---------------|
| `engine`            | Filtering engine to use, either "scipy" or "numba".                                        | `median`                       | str   | scipy         |
| `outlier_threshold` | The minimum difference abs(array - mean) for a pixel to be considered an outlier           | `distance`                     | float | 2             |
| `radius`            | The radius in which the average value is calculated                                        | `distance`                     | float | 5             |
| `sigma`             | The sigma of the Gaussian kernel                                                           | `gaussian`                     | float | 5             |
| `size`              | The size of the window to use (must be odd).                                               | `median`, `mean`, `min`, `max` | int   | 5             |
| `kwargs`            | Kwargs from [scipy](https://docs.scipy.org/doc/scipy/reference/ndimage.html) are available | `gaussian`, `min`, `max`       | dict  |               |

```{note}
The `median` filter can be computationally intensive, especially on large rasters. GeoUtils supports the use of
[Numba](https://numba.pydata.org/) to accelerate filter computations. To enable Numba, ensure it is installed in your
environment and set the `engine` parameter to `numba` when applying the filter
```

## Applying filters
Filters can be applied to a {class}`~geoutils.Raster` object using the {func}`~geoutils.Raster.filter` function.
For example:

```{code-cell} ipython3
import geoutils as gu
filename_rast = gu.examples.get_path("exploradores_aster_dem")
rast = gu.Raster(filename_rast)
# Filter the raster with a median filter of size 5
rast_filtered = rast.filter("median", size=5)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

import matplotlib.pyplot as plt

f, ax = plt.subplots(1, 2)
ax[0].set_title("Original Raster")
rast.plot(ax=ax[0])
ax[1].set_title("Filtered raster")
rast_filtered.plot(ax=ax[1])
plt.tight_layout()
```

For example, to apply a median filter:

```{code-cell} ipython3
# Apply a median filter with a kernel size of 3
filtered_raster = gu.filters.median_filter(rast.data, size=3)
```

Users can also apply custom filters by providing a function that takes a 2D numpy array as input and returns a filtered 2D numpy array.

```{code-cell} ipython3
import numpy as np

# Filter the raster with a hand-made filter
def double_filter(arr: np.ndarray) -> np.ndarray:
    return arr * 2
rast_double = rast.filter(double_filter)
```

## Reducer windows

Pass a {class}`~geoutils.operators.Reducer` to choose the calculation and a
{class}`~geoutils.operators.GridNeighbours` to choose the raster cells:

```{code-block} python
from geoutils.operators import GridNeighbours
from geoutils.operators.reducer import Mean

window = GridNeighbours(size=5, shape="circular")
smoothed = rast.filter(Mean(neighborhood=window), fractional=True)
```

`fractional=True` weights each cell by its covered area. It supports square and circular windows in pixel units.
Other neighborhoods can supply arbitrary row/column offsets. `size` and `kernel_shape` override a reducer's window
for that call. Without a configured window, filtering uses a 3 × 3 square.

Reducer filters preserve missing centers by default; `preserve_nodata=False` lets valid neighbors fill them.
`nodata_handling="propagate"` makes any missing neighbor invalidate the result, and `boundless=False` requires a
complete window. Bands are filtered independently, including with Dask and multiprocessing.

Built-in reductions share sliding sums and convolution with raster filters. Dense calls to
{meth}`~geoutils.Raster.resample_at_points` reuse these calculations when their windows have the same alignment;
sparse targets and custom reducers evaluate their neighborhoods directly. A custom reducer receives the same
values, coordinates, source IDs and optional area weights through either API.

Legacy named circular mean filters select cells strictly inside the radius. `GridNeighbours(size=5, shape="circular")` includes
cells whose centers lie on that boundary. Fractional circles use covered area instead of either center test.

## Point cloud neighborhood filters

{meth}`~geoutils.PointCloud.filter` replaces each active point value with a reduction of neighboring values selected
by horizontal X/Y distance. Built-in names include `"mean"`, `"median"`, `"min"`, `"max"`, `"range"`, `"count"`,
`"sum"`, `"std"` and `"rms"`; any {class}`~geoutils.operators.Reducer` can provide another calculation. A radius uses
CRS units, `k` selects the nearest points, and passing both first applies the radius and then the count limit. The
default includes the point being filtered. Set `include_self=False` for a leave-one-out calculation such as the
equivalent PDAL `filters.zsmooth` median.

Point rows, horizontal locations and other dataframe columns are preserved. Missing values are ignored by default;
use `nodata_propagation="propagate"` when any missing neighbor should make the output missing. `min_points` requires a
minimum number of finite neighbors. A point that does not meet that requirement receives a missing value.

```{code-block} python
from geoutils.operators.reducer import Median

filtered = points.filter(
    method=Median(),
    radius=2,
    include_self=False,
    min_points=3,
)
```

A reducer's {class}`~geoutils.operators.PointNeighbours` supplies its radius and nearest-neighbor limit when those
arguments are omitted. Explicit arguments override the configured limit without changing the reducer. An explicit
`radius=None` removes the distance limit; `k=None` removes the count limit. With no configured neighborhood or supplied
limits, the radius is one CRS unit.

Dask dataframes return a lazy point dataframe and require active values in a named column. Known spatial partition
bounds let Dask skip distant source partitions, so call `calculate_spatial_partitions()` before filtering a large
file-backed point cloud. Multiprocessing reads the radius-expanded bounds of each row partition and writes the result
to the file selected by {class}`~geoutils.multiproc.MultiprocConfig`. Storing nearby input points in the same
partitions reduces the size of those reads. Both Dask and multiprocessing require a finite radius.

For dense point clouds, `batch_size` limits how many target points build neighbor pairs at once. Smaller batches lower
temporary memory use at the cost of more query calls. For multiprocessing, `mp_config.chunks` separately controls how
many target rows each worker reads and writes as one partition.
