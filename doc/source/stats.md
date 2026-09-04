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

By default and without any specification, this function computes the following main statistics:
minimum, maximum, mean, standard deviation, normalized median absolute deviation, total count, and percentage of valid points.
```{code-cell} ipython3
rast.get_stats()
```

To compute all available statistics, set `stats_name` to `all`.
```{code-cell} ipython3
rast.get_stats("all")
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

## Grouped statistics

The {meth}`~geoutils.Raster.grouped_stats` and {meth}`~geoutils.PointCloud.grouped_stats` methods calculate one exact
grouping on the object's spatial support. Continuous variables are identified explicitly in `bins`, while discrete
variables use ordered `categories`. Raster bands, point columns, external spatial objects and aligned arrays can be
used as groupers.

The result is an ordinary {class}`pandas.DataFrame`. Its row index contains a {class}`pandas.IntervalIndex` or ordered
categories, and multiple groupers form a {class}`pandas.MultiIndex`. Its columns identify both the selected value and
statistic, which allows values with different missing data patterns to retain their own counts.

```{code-cell} ipython3
# Group the first raster band into ten elevation intervals
grouped = rast.grouped_stats(
    by={"elevation": 1},
    values={"elevation": 1},
    bins={"elevation": 10},
    statistics=("median", "nmad"),
)
grouped.head()
```

Pass exact numeric edges or a {class}`pandas.IntervalIndex` when boundaries or closure need to be controlled. Discrete
categories retain their supplied order. By default, combinations with no eligible locations are omitted; use
`observed=False` to include the complete declared product with zero counts.

Set `return_masks=True` to receive the dataframe and a mapping from every row key to a Boolean object on the same
support. The mapping stores one integer group layer and creates a Boolean mask only when a key is accessed. Raster
masks can therefore be written directly, while Xarray and point cloud calls retain their corresponding object type.

```{code-block} python
grouped, masks = rast.grouped_stats(
    by={"elevation": 1},
    bins={"elevation": [0, 1000, 2000, 4000, 8000]},
    return_masks=True,
)
first_key = grouped.index[0]
masks[first_key].to_file("first_elevation_bin.tif")
```

Masks describe complete bin or category membership after the user mask and grouper validity. They intentionally do
not apply selected value missing data or random subsampling, so the finite count for a value can be lower than the
number of true cells in its group mask.

Use {func}`~geoutils.stats.grouped_stats` directly for aligned arrays. The
{func}`~geoutils.stats.plot_grouped_stats` helper draws the statistic with sample counts for one or two group
dimensions. Exact marginal statistics remain separate `grouped_stats` calls because they cannot generally be derived
from summaries of a higher dimensional grouping.

## Cosampling

The {meth}`~geoutils.Raster.cosample` and {meth}`~geoutils.PointCloud.cosample` methods extract two primary datasets
at the same finite locations. Optional auxiliary variables can follow either primary input. A point cloud provides
the common support automatically when one is present; otherwise the calling raster grid is used.

The returned {class}`~geoutils.CoSampleResult` contains bounded eager arrays, selected coordinates and indexes into the
original support. It can be converted to a point cloud or expanded with NaN outside the sample for procedural code.

```{code-cell} ipython3
# Keep 10,000 locations that are valid in both rasters
paired = rast.cosample(
    rast + 1,
    subsample=10_000,
    random_state=42,
)
paired.to_pointcloud(self_name="elevation", other_name="shifted_elevation")
```

Mismatched grids raise by default; pass `align="reproject"` to make resampling explicit. The default `topk` strategy
selects the same raster cells for a fixed seed regardless of Dask chunk layout. On point support, invalid point values
and raster validity are evaluated before sampling, while raster values are interpolated only at selected points.

## Variograms

Install `geoutils[geostat]` to estimate and fit variograms with SciKit-GStat model and estimator functions. The public
{class}`~geoutils.Variogram` is a lightweight record: it keeps lag boundaries, empirical semivariance, sampling error,
pair count and optional fitted parameters. It does not retain sampled pairs or a SciKit-GStat object.

```{code-block} python
variogram = rast.variogram(
    n_pairs=1_000_000,
    n_lags=24,
    n_runs=3,
    model=["spherical", "gaussian"],
    random_state=42,
)

# Tabular or labelled interchange without backend state
table = variogram.to_dataframe()
dataset = variogram.to_xarray()
portable = variogram.to_dict()
```

Raster pairs and lag bins are sampled logarithmically so structure at short distances remains represented without
forming a full pairwise matrix. Point clouds expose the same interface with strategies for irregular coordinates.
Advanced users can inspect or reuse the original pair sample separately:

```{code-block} python
pairs = rast.sample_pairs(
    n_pairs=1_000_000,
    strategy="chunk_anchors",
    hybrid_local_fraction=0.5,
    random_state=42,
)
empirical = gu.Variogram.from_pairs(pairs, estimator="dowd", bins="log")
fitted = empirical.fit(["spherical", "gaussian"])
```

Known fitted parameters can also be represented directly and converted for downstream covariance workflows:

```{code-block} python
model = gu.Variogram.from_model(
    "gaussian",
    effective_range=500,
    partial_sill=4,
    nugget=0.2,
)
gstools_model = model.to_gstools(dim=2).model

# The nugget is returned separately for use as likelihood noise
gpytorch_conversion = model.to_gpytorch()
```
