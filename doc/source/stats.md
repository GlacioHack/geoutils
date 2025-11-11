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
rast = gu.Raster(filename_rast)
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

## Subsampling

The {func}`~geoutils.Raster.subsample` method allows to efficiently extract a valid random subsample from a raster or a point cloud. It can conveniently
return the output as a point cloud, or as an array.

The subsample size can be defined either as a fraction of valid values (floating value strictly between 0 and 1), or as a number of samples (integer value
above 1).

```{code-cell} ipython3
# Subsample 10% of the raster valid values
rast.subsample(subsample=0.1)
```
