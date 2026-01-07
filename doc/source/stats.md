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
(stats)=

# Statistics

GeoUtils supports statistical analysis tailored to geospatial objects.

For a {class}`~geoutils.Raster` or a {class}`~geoutils.PointCloud`, the statistics are naturally performed on the {attr}`~geoutils.Raster.data` attribute
which is clearly defined.

[//]: # (For a {class}`~geoutils.Vector`, statistics have to be performed on a specific column.)

```{warning}
The API for statistical features is preliminary and might change with the release of zonal and grouped statistics.
```

## Estimators

The {func}`~geoutils.Raster.get_stats` method allows to extract key statistical estimators from a raster or a point cloud, optionally subsetting to an
inlier mask.

Supported statistics are :
- **Mean:** arithmetic mean of the data, ignoring masked values.
- **Median:** middle value when the valid data points are sorted in increasing order, ignoring masked values.
- **Max:** maximum value among the data, ignoring masked values.
- **Min:** minimum value among the data, ignoring masked values.
- **Sum:** sum of all data, ignoring masked values.
- **Sum of squares:** sum of the squares of all data, ignoring masked values.
- **90th percentile:** point below which 90% of the data falls, ignoring masked values.
- **IQR (Interquartile Range):** difference between the 75th and 25th percentile of a dataset, ignoring masked values.
- **LE90 (Linear Error with 90% confidence):** difference between the 95th and 5th percentiles of a dataset, representing the range within which 90% of the data points lie. Ignore masked values.
- **NMAD (Normalized Median Absolute Deviation):** robust measure of variability in the data, less sensitive to outliers compared to standard deviation. Ignore masked values.
- **RMSE (Root Mean Square Error):** commonly used to express the magnitude of errors or variability and can give insight into the spread of the data. Only relevant when the raster represents a difference of two objects. Ignore masked values.
- **Std (Standard deviation):** measures the spread or dispersion of the data around the mean, ignoring masked values.
- **Valid count:** number of finite data points in the array. It counts the non-masked elements.
- **Total count:** total size of the raster.
- **Percentage valid points:** ratio between **Valid count** and **Total count**.

If an inlier mask is passed:
- **Total inlier count:** number of data points in the inlier mask.
- **Valid inlier count:** number of unmasked data points in the array after applying the inlier mask.
- **Percentage inlier points:** ratio between **Valid inlier count** and **Valid count**. Useful for classification statistics.
- **Percentage valid inlier points:** ratio between **Valid inlier count** and **Total inlier count**.

Callable functions are supported as well.

```{code-cell} ipython3
import geoutils as gu
import numpy as np

# Instantiate a raster from a filename on disk
filename_rast = gu.examples.get_path("exploradores_aster_dem")
rast = gu.Raster(filename_rast, force_nodata=-9999)
rast
```

Get all default statistics:
```{code-cell} ipython3
rast.get_stats()
```

Get a single statistic (e.g., 'mean') as a float:
```{code-cell} ipython3
rast.get_stats("mean")
```

Get multiple statistics:
```{code-cell} ipython3
rast.get_stats(["mean", "max", "std"])
```

Using a custom callable statistic:
```{code-cell} ipython3
def custom_stat(data):
    return np.nansum(data > 100)  # Count the number of pixels above 100
rast.get_stats(custom_stat)
```

Passing an inlier mask:
```{code-cell} ipython3
inlier_mask = rast > 1500
rast.get_stats(inlier_mask=inlier_mask)
```

## Grouped statistics

GeoUtils provides support for grouped statistics, allowing statistics to be computed independently over subsets of data
defined by one or more grouping bins. This is particularly useful when analyzing how statistical properties vary
across classes, bins, or segmentation derived from the data itself.


### Example with altitude intervals
In this example, we will create different altitude classes from a chosen interval [400, 1000, 2000, 3000, >3000].
Once these bins are created, we reapply them and compute the mean, minimum, and maximum values of the same raster
for each sub-interval. It is also possible to use a reference other than the raster itself for the group_by.

A dictionary containing the masks that have been created during the computation will also be returned by the function.
Using geoUtils functions makes it very easy to visualise them.

```{code-cell} ipython3
from geoutils.stats import grouped_stats
import math
import matplotlib.pyplot as plt

group_by = {"rast": rast}
bins = {"rast": [400, 1000, 2000, 3000, np.inf]}
to_aggregate = {"rast": rast}
statistics = ["mean", "min", "max"]

df, masks = grouped_stats.grouped_stats(group_by, bins, to_aggregate, statistics)
df
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

groups = list(masks["groupby_rast"].keys())
n = len(groups)

ncols = 3
nrows = math.ceil(n / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
axes = axes.flatten()

for ax, group in zip(axes, groups):
    masks["groupby_rast"][group].plot(ax=ax)
    ax.set_title(group)

for ax in axes[n:]:
    ax.axis("off")

plt.tight_layout()
```

```{warning}
Bins can be presented in different ways. It is possible to integrate an interval of minimum 2 values, a mask or a
segmentation map in raster format.
```


### Example with altitude masks

In this example, we will create a mask such as altitude is more than 2000 meters.
Once these masks are created as a raster, we reapply them and compute the mean, minimum, and maximum values of the same raster
for masks = True. It is also possible to use a reference other than the raster itself for the group_by.

```{code-cell} ipython3
from geoutils.stats import grouped_stats

group_by = {"rast": rast}
elev_mask = rast > 2000
bins = {"rast": elev_mask}

to_aggregate = {"rast": rast}
statistics = ["mean", "min", "max"]

df, _ = grouped_stats.grouped_stats(group_by, bins, to_aggregate, statistics)
df
```

```{code-cell} ipython3
elev_mask.plot()
plt.show()
```
## Subsampling

The {func}`~geoutils.Raster.subsample` method allows to efficiently extract a valid random subsample from a raster or a point cloud. It can conveniently
return the output as a point cloud, or as an array.

The subsample size can be defined either as a fraction of valid values (floating value strictly between 0 and 1), or as a number of samples (integer value
above 1).

```{code-cell} ipython3
# Subsample 10% of the raster valid values
rast.subsample(subsample=0.1)
```
