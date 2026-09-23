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
GeoUtils implements **filtering capabilities for regular and irregular data** with **scalable execution** on large datasets and **geospatial awareness** to define filter distances and skip invalid values.

Filters do not include only methods aimed at removing outliers (e.g. Gaussian or median), but can be any computation on a
window (for rasters) or neighborhood (for points), such as deriving a **local derivative** or a **local autocorrelation index**.

```{code-cell} ipython3
:tags: [remove-cell]

# Match the figure resolution and text size used by the other feature pages
from matplotlib import pyplot as plt
plt.rcParams["figure.dpi"] = 600
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["font.size"] = 9
```

## Raster filters

{meth}`ds.rst.filter() or Raster.filter() <RasterBase.filter>`

Raster filters are optimized for speed can be applied directly to a raster:

```{code-cell} ipython3
---
mystnb:
  output_stderr: remove
---
import geoutils as gu
filename_rast = gu.examples.get_path("exploradores_aster_dem")
ds = gu.open_raster(filename_rast)
# Filter the raster with a median filter of size 15
ds_filtered = ds.rst.filter("median", size=15)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

f, ax = plt.subplots(1, 2)
ax[0].set_title("Original raster")
ds.rst.plot(ax=ax[0], cmap="terrain", cbar_title="Elevation (m)")
ax[1].set_title("Filtered raster")
ds_filtered.rst.plot(ax=ax[1], cmap="terrain", cbar_title="Elevation (m)")
_ = ax[1].set_yticklabels([])
plt.tight_layout()
```

### Available methods

The following filtering methods are available:

| Filter Name | Description                                                                                                                                                           | Typical Effect                                                  |
|:------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------|
| `gaussian`  | Applies a Gaussian (blur) filter with a specified sigma.                                                                                                              | Smooths the image, reduces noise while slightly blurring edges. |
| `median`    | Applies a median filter over a sliding window.                                                                                                                        | Reduces noise while preserving edges better than Gaussian.      |
| `mean`      | Applies a mean (average) filter with a specified kernel size.                                                                                                         | Smooths the image uniformly, reduces high-frequency noise.      |
| `max`       | Applies a maximum filter over a sliding window.                                                                                                                       | Enhances bright regions, expands high-intensity areas.          |
| `min`       | Applies a minimum filter over a sliding window.                                                                                                                       | Suppresses bright regions, expands dark regions. |
| `distance`  | Removes pixels that deviate strongly from local neighborhood average (within a radius).                                                                               | Removes outliers and anomalous values based on local context.   |
| `custom`    | Allows users to define their own filter function to be applied to a numpy array.                                                                                      |                                                                 |

### Parameters

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

### Using a custom filter

Custom filters can be provided as functions that take a 2D NumPy array and return a filtered 2D NumPy array.
For example, combine the horizontal and vertical elevation gradients to derive terrain slope:

```{code-cell} ipython3
import numpy as np

# Convert the elevation gradient in each map direction to slope in degrees
pixel_width, pixel_height = ds.rst.res
def slope_filter(arr: np.ndarray) -> np.ndarray:
    gradient_y, gradient_x = np.gradient(arr, pixel_height, pixel_width)
    return np.degrees(np.arctan(np.hypot(gradient_x, gradient_y)))

ds_slope = ds.rst.filter(slope_filter)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

f, ax = plt.subplots(1, 2)
ax[0].set_title("Original raster")
ds.rst.plot(ax=ax[0], cmap="terrain", cbar_title="Elevation (m)")
ax[1].set_title("Terrain slope")
ds_slope.rst.plot(ax=ax[1], cmap="magma", cbar_title="Slope (°)", vmin=0, vmax=60)
_ = ax[1].set_yticklabels([])
plt.tight_layout()
```
