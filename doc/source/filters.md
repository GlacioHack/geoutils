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
GeoUtils implements **filtering capabilities for both rasters and point clouds** .

All support **scalable execution** on large datasets and **geospatial awareness** to define filter distances and skip invalid values.

Filters are not limited to smoothing data or removing outliers (e.g. Gaussian or median filters). They can perform any 
computation on a window (for rasters) or neighborhood (for points), such as deriving a **local derivative** or a **local autocorrelation index**.

```{seealso}
To derive **terrain attributes** on elevation data, which are also a type of filter, see [xDEM's implementations](https://xdem.readthedocs.io/en/stable/terrain.html) based on peer-reviewed studies.
```

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

A raster filter **computes each output pixel value from nearby input pixels**, returning a new raster with the same shape and metadata.
Neighborhoods are defined with a **window size and shape**, and optionally support fractional cell coverage.

For example, apply a median filter in a **15 × 15 pixel window**:

```{code-cell} ipython3
---
mystnb:
  output_stderr: remove
---
import geoutils as gu
filename_rast = gu.examples.get_path("exploradores_aster_dem")
ds = gu.open_raster(filename_rast)
# Filter the raster with a median filter of size 15
ds_filtered = ds.rst.filter(method="median", backend="scipy", window=15)
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

Use `window` to set the window size in pixels (odd, three by default), `sigma` sets the Gaussian standard deviation
in pixels. Filters use SciPy by default, use `backend="numba"` to use Numba.

(filters-points)=
## Point filters

{meth}`points.pc.filter() or PointCloud.filter() <PointCloudBase.filter>`

A point filter **computes each output point value from nearby input values**, returning a new point cloud with the same coordinates and metadata.
Neighborhoods are defined by a **radius** in coordinate units and **maximum closest number of points**.

For example, to compute the mean in a 1000-m radius, using at most the closest 20 points:

```{code-cell} ipython3
# Sample the projected DEM, then average up to 20 measurements within 1000 m
points = ds.rst.subsample(1500, random_state=42)
smoothed_points = points.pc.filter(
    method="mean", backend="scipy", radius=1000, max_points=20
)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

# Compare the same points with a shared extent and color scale
vmin, vmax = points.pc.data.min(), points.pc.data.max()
f, ax = plt.subplots(1, 2)
ax[0].set_title("Original points")
points.pc.plot(
    ref=ds, ax=ax[0], cmap="terrain", cbar_title="Elevation (m)",
    markersize=5, vmin=vmin, vmax=vmax
)
ax[1].set_title("Filtered points")
smoothed_points.pc.plot(
    ref=ds, ax=ax[1], cmap="terrain", cbar_title="Elevation (m)",
    markersize=5, vmin=vmin, vmax=vmax
)
_ = ax[1].set_yticklabels([])
plt.tight_layout()
```

Use `include_self=False` to exclude the point original value at this coordinate, or `min_points` to require enough valid neighbors.
Lower `batch_size` to reduce temporary memory for dense neighborhoods.

## Available methods

GeoUtils supports the following raster and point filters. See {ref}`transformations-reducers`
for illustrations of supported reducers.

```{seealso}
Refer to {ref}`core-operators` for more details and customization of reducers and neighborhoods.
```

| Method | Description                                           | Raster | Point |
|:-------|:------------------------------------------------------|:------:|:-----:|
| `count` | Number of valid neighbors                             | ✓ | ✓ |
| `sum` | Sum of neighboring values                             | ✓ | ✓ |
| `mean`, `average` | Mean of neighboring values                            | ✓ | ✓ |
| `min`, `minimum` | Minimum value                                         | ✓ | ✓ |
| `max`, `maximum` | Maximum value                                         | ✓ | ✓ |
| `range` | Difference between maximum and minimum                | ✓ | ✓ |
| `rms`, `rmse` | Root mean square of neighboring values                | ✓ | ✓ |
| `stdev`, `std`, `standard_deviation` | Population standard deviation                         | ✓ | ✓ |
| `median`, `med` | Median of neighboring values                          | ✓ | ✓ |
| `mode` | Most frequent value                                   | ✓ | ✓ |
| `q1`, `q3` | First or third quartile                               | ✓ | ✓ |
| `quantile` | Quantile at the supplied fraction                     | ✓ | ✓ |
| `average_distance` | Mean distance from neighbors to the target            | ✓ | ✓ |
| `average_distance_pts` | Mean distance between pairs of neighbors              | ✓ | ✓ |
| `gaussian` | Smooth with a Gaussian kernel                         | ✓ | — |
| `distance` | Mask outliers relative to a Gaussian-smoothed average | ✓ | — |
| {class}`~geoutils.operators.Reducer` | Custom calculation on each neighborhood               | ✓ | ✓ |
| Callable | Custom function taking and returning a 2D NumPy array | ✓ | — |

## Customizing filters

### Nodata propagation

Filters can **use the available values around gaps** or **propagate missing values to the output**.
Below, we give an example of both behaviors by applying a mean filter to a DEM with gaps:

```{code-cell} ipython3
import numpy as np
from geoutils.operators.reducer import Mean

# Add a gap in a copy of the DEM detail
gap_dem = ds.rst.crop((630000, 4839000, 633000, 4842000))
gap_values = gap_dem.values.copy()
gap_values[45:55, 45:55] = np.nan
gap_dem = gap_dem.rst.copy(new_array=gap_values)

# Compare nodata rules, allowing originally missing centers to receive a value
ignored = gap_dem.rst.filter(
    Mean(), window=15, nodata_handling="ignore", preserve_nodata=False
)
propagated = gap_dem.rst.filter(
    Mean(), window=15, nodata_handling="propagate", preserve_nodata=False
)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

# Use the same color scale and show missing cells in gray
elevation_min, elevation_max = np.nanmin(gap_values), np.nanmax(gap_values)
cmap = plt.colormaps["terrain"].with_extremes(bad="lightgray")
f, ax = plt.subplots(1, 3, figsize=(9, 3.5))
for axis, raster, title in zip(
    ax, [gap_dem, ignored, propagated],
    ["Input with gaps", 'nodata_handling =\n"ignore"', 'nodata_handling =\n"propagate"']
):
    raster.rst.plot(
        ax=axis, title=title, cmap=cmap, vmin=elevation_min, vmax=elevation_max
    )
plt.tight_layout()
```

Gray areas show missing values.

```{tip}
To **calculate values at missing raster pixels**, set `preserve_nodata=False`.
See {ref}`core-nodata` for more on nodata propagation in raster and point filters.
```

### Custom point reducers

A {ref}`custom reducer <core-custom-operators>` lets you **define a calculation on neighboring values**,
while GeoUtils handles their selection and missing data.

Below, we define a **trimmed mean**, which discards the lowest and highest values before averaging.
We apply it to a point cloud with a few added outliers to illustrate how it reduces their influence:

```{code-cell} ipython3
from geoutils.operators import LocalData, Reducer


class TrimmedMean(Reducer):
    """Average neighboring values after discarding one minimum and one maximum."""

    minimum_inputs = 3

    def reduce(self, data: LocalData) -> float:
        """Calculate the trimmed mean from valid values in one neighborhood."""
        # Remove the two extreme values without sorting the neighborhood
        values = data.values
        trimmed_sum = np.sum(values) - np.min(values) - np.max(values)
        return float(trimmed_sum / (values.size - 2))


# Add a few positive outliers to a copy of the sampled elevations
noisy_values = np.array(points.pc.data, dtype=float, copy=True)
noisy_values[::100] += 2000
noisy_points = points.pc.copy(new_array=noisy_values)
trimmed_points = noisy_points.pc.filter(
    TrimmedMean(), radius=1000, max_points=20, batch_size=256
)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

# Compare the noisy input and filtered values with the same color scale
vmin, vmax = noisy_values.min(), noisy_values.max()
f, ax = plt.subplots(1, 2)
ax[0].set_title("Points with outliers")
noisy_points.pc.plot(
    ref=ds, ax=ax[0], cmap="terrain", cbar_title="Elevation (m)",
    markersize=5, vmin=vmin, vmax=vmax
)
ax[1].set_title("Trimmed mean")
trimmed_points.pc.plot(
    ref=ds, ax=ax[1], cmap="terrain", cbar_title="Elevation (m)",
    markersize=5, vmin=vmin, vmax=vmax
)
_ = ax[1].set_yticklabels([])
plt.tight_layout()
```

The same reducer also supports **lazy, chunked execution with Dask**:

```{code-cell} ipython3
# Split the points into chunks of 500 and build a lazy filtering calculation
chunked_points = noisy_points.pc.to_xarray().chunk({"point": 500})
lazy_trimmed = chunked_points.pc.filter(
    TrimmedMean(), radius=1000, max_points=20, batch_size=256
)
```

```{tip}
Use a **finite radius** for chunked point filters so each chunk can read its surrounding neighbors.
See {ref}`scalability-support` for details on chunked execution.
```

### Custom raster functions

Raster filters also accept **custom functions operating on a NumPy array**.
Below, we derive **terrain slope** by combining the horizontal and vertical elevation gradients:

```{code-cell} ipython3
import numpy as np

# Convert the elevation gradient in each map direction to slope in degrees
pixel_width, pixel_height = ds.rst.res


def slope_filter(arr: np.ndarray) -> np.ndarray:
    """Calculate terrain slope in degrees from the elevation gradients."""
    gradient_y, gradient_x = np.gradient(arr, pixel_height, pixel_width)
    return np.degrees(np.arctan(np.hypot(gradient_x, gradient_y)))


ds_slope = ds.rst.filter(slope_filter, window=3)
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

```{tip}
When filtering in chunks, include the **neighboring pixels needed by your function** in `window`.
Here, the gradient uses one pixel on either side, so `window=3` is sufficient.
```