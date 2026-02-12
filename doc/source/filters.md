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
GeoUtils provides several filters to process raster data. These filters can be found in the `geoutils.filters` module.
They can be applied to {class}`~geoutils.Raster` objects.

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
`median` filter can be computationally intensive, especially on large rasters. GeoUtils supports the use of
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

Another way to apply filters is through the {func}`~geoutils.filters.filter_name` methods directly on the raster data.

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
