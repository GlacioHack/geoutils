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

GeoUtils provides implementations to derive **statistical estimators relevant to geospatial data** on
**rasters** and **point clouds**, with efficient reduction for large datasets.

Several types of operations are supported:

- {ref}`stats-global-grouped` derive **distributional estimators by bins or categories of other variables**,
- {ref}`stats-spatial` derives **spatial correlation estimators** as a function of distance.

**Zonal statistics are included** as a form of grouped statistics where vector features define the categories.

Any custom estimator can be passed, with certain estimators listed below **optimized for
larger-than-memory samples** (both for grouped and spatial estimators).

```{note}
Statistical operations **accept datasets with different CRS or spatial support** (e.g. binning a raster with a point cloud), using efficient co-sampling techniques under-the-hood.
For more details on those, see the **{ref}`sampling` feature page**.

All methods on this page accept combinations of **out-of-memory rasters, point clouds and vectors**. They normally
follow the chunks or partitions used when opening each input; see {ref}`stats-chunked-strategies` for advanced controls.
```

```{code-cell} ipython3
:tags: [remove-cell]

# Match the figure resolution and text size used by the other feature pages
from matplotlib import pyplot as plt
plt.rcParams["figure.dpi"] = 600
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["font.size"] = 9
```

```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the code for opening example files"
:  code_prompt_hide: "Hide the code for opening example files"

import geoutils as gu
import numpy as np
import pandas as pd

# Open an elevation raster and glacier outlines covering the same region
ds = gu.open_raster(gu.examples.get_path("exploradores_aster_dem"))
glaciers = gu.open_vector(gu.examples.get_path("exploradores_rgi_outlines"))
```

(stats-global-grouped)=
## Global and grouped statistics

Global statistics derive **estimators on the entire raster or point cloud**, while grouped statistics apply the
same **estimators to continuous bins or discrete categories**, with potentially several groupings at once.

(stats-estimators)=
### Estimators

GeoUtils accepts **any custom estimator**, and provides the built-ins below.

"Mergeable" estimators can successively aggregate values per chunk to exactly derive the final value, never
requiring to hold the whole sample at once. Others require the group in memory.

| Estimator | Meaning                                                | Mergeable |
| --- |--------------------------------------------------------| --- |
| Mean | Arithmetic average                                     | **Yes** |
| Standard deviation | Spread around the mean                                 | **Yes** |
| Min | Smallest value                                         | **Yes** |
| Max | Largest value                                          | **Yes** |
| Sum, sum of squares | Sum of values or their squares                         | **Yes** |
| RMSE | Square root of the mean squared value                  | **Yes** |
| Counts and percentages | Valid and total counts, and their relative proportions | **Yes** |
| Median | Middle value                                           | **No** |
| NMAD | Normalized median absolute deviation ("median-STD")     | **No** |
| Percentiles, IQR, LE90 | Quantiles and ranges derived from them                 | **No** |
| Custom callable | User-defined calculation                               | **No** |


```{tip}
To **avoid holding a whole group in memory** with a non-mergeable estimator (including a custom function),
set a maximum number of samples through `subsample`, with `subsample_per_group=True`.
This commonly preserves sufficient information because nearby geospatial observations are often highly correlated.
```

(stats-global)=
### Global statistics

{meth}`ds.rst.stats() or Raster.stats() <RasterBase.stats>` with `by=None`<br>
{meth}`gdf.pc.stats() or PointCloud.stats() <PointCloudBase.stats>` with `by=None`<br>

Global statistics allow to derive **any statistical estimator on any or all raster bands or point cloud columns**.

By default, the mean, standard deviation, max, min and count are computed (**none of which requires holding large samples** in memory; see {ref}`stats-estimators`).
Nodata values are skipped by default, but explicitly reported in value counts.


```{code-cell} ipython3
# Compute the default summary statistics
ds.rst.stats()
```

Requesting **one statistic returns a scalar**, and requesting **several statistics returns a dictionary**. Use `"all"` to request every
available estimator.

```{code-cell} ipython3
ds.rst.stats("mean")
```

```{code-cell} ipython3
ds.rst.stats(["mean", "median", "std", "nmad"])
```

Custom estimators receive the data array and must handle its NaNs (with Xarray accessor) or NumPy masked array (with a {class}`~geoutils.Raster`), for example:

```{code-cell} ipython3
def count_high_elevations(data: np.ndarray) -> int:
    """Count observations above 1500 m in the selected elevation data."""

    # Count only finite observations
    values = np.ma.asarray(data).compressed()
    return int(np.count_nonzero(np.isfinite(values) & (values > 1500)))

ds.rst.stats(count_high_elevations)
```

(stats-global-masking)=
#### Masking

Use `mask` to **restrict the statistics to selected locations**, which accepts vector or raster/point-cloud/array mask:

```{code-cell} ipython3
# Summarize elevations inside all glacier outlines together
ds.rst.stats(["mean", "std", "valid count"], mask=glaciers)
```

(stats-global-subsampling)=
#### Subsampling

Use `subsample` to cap the sample size, which is especially useful with **robust estimators**, such as the median and NMAD,
that are less sensitive to outliers but need to hold the whole (see {ref}`stats-estimators`):

```{code-cell} ipython3
ds.rst.stats(["mean", "median", "std", "nmad"], subsample=100_000)
```

(stats-global-values)=
#### Multiple values

Multi-band rasters or multi-column point clouds can be summarized at once. Use `values` to select the data: raster
bands (starting at one), or point cloud column names; and **name them for the output dictionary** by using a mapping:

```{code-cell} ipython3
# Name and summarize two bands of an RGB image
rgb = gu.open_raster(gu.examples.get_path("everest_landsat_rgb"))
# Select band 1/3, naming them "red" and "blue" for the output
rgb.rst.stats(["mean", "std"], values={"red": 1, "blue": 3})
```

(stats-grouped)=
### Grouped statistics

{meth}`ds.rst.stats() or Raster.stats() <RasterBase.stats>` with ``by``<br>
{meth}`gdf.pc.stats() or PointCloud.stats() <PointCloudBase.stats>` with ``by``<br>

Grouped statistics derive **statistical estimators within bins or categories of one or more variables**. Bins can be
continuous intervals, such as elevation bands, while categories can be land-cover classes or vector zones.

Use `by` to name the grouping variables. Vector attributes can provide numeric grouping variables; supply `bins` to
treat an attribute as continuous.

{ref}`Masking <stats-global-masking>`, {ref}`subsampling <stats-global-subsampling>` and
{ref}`selecting multiple values <stats-global-values>` work as for global statistics. Set `subsample_per_group=True`
to apply the subsample independently within each group.

The examples below summarize three RGB bands by elevation and glacier coverage. The RGB rendering, elevation raster,
and glacier outlines share the same grid and location:

```{code-cell} ipython3
:tags: [hide-cell]

# Create a three-band terrain rendering on the elevation grid
elevation_values = np.asarray(ds)
valid_elevation = np.isfinite(elevation_values)
lower, upper = np.nanpercentile(elevation_values, [2, 98])
normalized_elevation = np.clip((elevation_values - lower) / (upper - lower), 0, 1)
terrain_rgb_values = plt.get_cmap("terrain")(normalized_elevation)[..., :3]
terrain_rgb_values[~valid_elevation] = np.nan
terrain_rgb = gu.RasterAccessor.from_array(
    np.moveaxis(terrain_rgb_values, -1, 0),
    transform=ds.rst.transform,
    crs=ds.rst.crs,
)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

fig, axes = plt.subplots(1, 2)
terrain_rgb.rst.plot(ax=axes[0])
glaciers.vct.plot(ref=ds, ax=axes[0], fc="none", ec="k", linewidth=0.6)
axes[0].set_title("Terrain RGB and glaciers")
ds.rst.plot(ax=axes[1], cmap="terrain", cbar_title="Elevation (m)")
glaciers.vct.plot(ref=ds, ax=axes[1], fc="none", ec="k", linewidth=0.6)
axes[1].set_title("Elevation and glaciers")
axes[1].set_yticklabels([])
plt.tight_layout()
```

#### Continuous bins

Use `bins` to define **intervals of a continuous variable**. For example, summarize the RGB bands within elevation
bands:

```{code-cell} ipython3
# Summarize all three color bands within explicit elevation intervals
elevation_edges = [0, 1000, 2000, 3000, 4000]
elevation_stats = terrain_rgb.rst.stats(
    ("mean", "std"),
    by={"elevation": ds},
    values={"red": 1, "green": 2, "blue": 3},
    bins={"elevation": elevation_edges},
)
elevation_stats
```

The output table has **one row per group and one column per value–statistic combination**. Each value also has a
finite `count`.

An integer, such as `bins={"elevation": 10}`, creates ten equal-width bins. Numeric edges include each lower edge
and the final upper edge. Pass a {class}`pandas.IntervalIndex` to choose which edges are included:

```{code-cell} ipython3
# Assign boundary elevations to the interval ending at that elevation
right_closed = pd.IntervalIndex.from_breaks(elevation_edges, closed="right")
terrain_rgb.rst.stats(
    "mean",
    by={"elevation": ds},
    values={"red": 1, "green": 2, "blue": 3},
    bins={"elevation": right_closed},
)
```

#### Discrete categories

Use `categories` to define **discrete classes and their order**. A boolean glacier mask, for example, separates terrain
inside and outside the outlines:

```{code-cell} ipython3
# Compare RGB values inside and outside glaciers
glacier_mask = glaciers.vct.create_mask(ds)
glacier_stats = terrain_rgb.rst.stats(
    ("mean", "std"),
    by={"glacier": glacier_mask},
    values={"red": 1, "green": 2, "blue": 3},
    categories={"glacier": [False, True]},
)
glacier_stats
```

The same operation applies to land-cover codes or other categorical variables. Values outside the declared categories
are excluded. Boolean and Pandas categorical inputs can supply their categories without an explicit declaration.

(stats-zonal)=
#### Zonal statistics

**Zonal statistics assign vector feature attributes as categories.** Pass `(vector, "column")` in `by`; the
geometries determine each zone, and the selected column provides its category. **A unique feature ID gives one
category per feature**, while repeated IDs combine several features into one category. Categories are inferred from
the column, so no `bins` or `categories` argument is needed.

For example, the glacier inventory's `RGIId` column identifies each glacier:

```{code-cell} ipython3
# Calculate separate RGB statistics for each glacier outline
glacier_zonal_stats = terrain_rgb.rst.stats(
    ("mean", "std"),
    by={"glacier": (glaciers, "RGIId")},
    values={"red": 1, "green": 2, "blue": 3},
    observed=False,
)
glacier_zonal_stats.sort_values(("red", "count"), ascending=False).head()
```

Locations outside all zones are excluded, and each statistic uses the finite values in its category. `observed=False`
retains categories without observations, with zero counts and undefined statistics.

The same grouping applies to point measurements:

```{code-cell} ipython3
# Summarize a sample of point elevations within each glacier
points = ds.rst.to_pointcloud(data_column_name="elevation", subsample=2000, random_state=42)
points.pc.stats(("mean", "std"), by={"glacier": (glaciers, "RGIId")}).head()
```

Raster cells are assigned by their centres, and points by their coordinates. Grouping assigns each location to one
category; use non-overlapping zones for independent feature statistics. Vector and GeoDataFrame inputs are both
accepted, including when the raster values are backed by Dask.

```{note}
`by={"zone": (zones, "id")}` groups by feature IDs. A bare vector, `by={"inside": zones}`, groups the union of all
features into inside/outside categories. `mask=zones` instead restricts all groups to locations inside the features.
```

#### Several grouping variables

**Several grouping variables define their joint bins or categories**. For example, summarize the RGB bands jointly
by elevation interval and glacier membership:

```{code-cell} ipython3
# Retain empty combinations so the complete comparison grid remains visible
joint_stats = terrain_rgb.rst.stats(
    ("mean", "std"),
    by={"elevation": ds, "glacier": glacier_mask},
    values={"red": 1, "green": 2, "blue": 3},
    bins={"elevation": elevation_edges},
    categories={"glacier": [False, True]},
    observed=False,
)
joint_stats.head()
```

The rows form a {class}`pandas.MultiIndex`. Vector zones can be combined with continuous bins in the same way,
by including both variables in `by`. `observed=False` retains empty combinations; the default, `observed=True`,
omits combinations without eligible locations.

Use {func}`~geoutils.stats.plot_grouped_stats` to **plot a statistic and its sample counts** for one or two grouping
variables. `min_count` hides estimates based on too few observations.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

# Display the mean red value and the number of observations supporting it
fig, ax = plt.subplots(figsize=(7, 4))
fig.subplots_adjust(right=0.78)
axes = gu.stats.plot_grouped_stats(joint_stats, value="red", statistic="mean", min_count=100, ax=ax)
axes["statistic"].set(xlabel="Elevation (m)", ylabel="Inside glacier")

# Place the color scale beyond the count panels so every elevation interval stays visible
axes["colorbar"].set_position((0.84, 0.11, 0.025, 0.5))
_ = axes["colorbar"].set_ylabel("Mean red value")
```

To summarize only one variable, make a separate grouping call. Statistics such as medians cannot generally be
combined from subgroup results.

#### Returning group masks

Set `return_masks=True` when the geographic membership of each group is also needed. The same row key retrieves its
mask:

```{code-cell} ipython3
_, elevation_masks = terrain_rgb.rst.stats(
    "mean",
    by={"elevation": ds},
    values={"red": 1},
    bins={"elevation": elevation_edges},
    return_masks=True,
)
first_interval = next(iter(elevation_masks))
first_mask = elevation_masks[first_interval]
first_interval
```

(stats-spatial)=
## Spatial statistics (variography)

Spatial statistics allows to describe **how differences between values change with distance (i.e. spatial autocorrelation).**

GeoUtils performs the **pair sampling and lag-bin reduction efficiently**, then relies on
[SciKit-GStat's semivariance estimators](https://scikit-gstat.readthedocs.io/en/latest/reference/estimator.html) to
derive estimators and fit models. Models can then be passed to other GeoUtils functions or exported to covariance
formats such as GSTools or GPyTorch for kriging, random field generation and other analyses.

(stats-spatial-estimators)=
### Estimators

 For mergeable estimators listed below, we also implement a specific aggregation routine
to derive them without holding more than a chunk in memory.

| Estimator | Meaning | Mergeable |
| --- | --- | --- |
| Matheron | Half the mean squared pair difference | **Yes** |
| Cressie–Hawkins | Robust transform of the mean square-root difference | **Yes** |
| MinMax | Difference between the maximum and minimum, normalized by the mean | **Yes** |
| Dowd | Robust median-based semivariance | **No** |
| Genton | Robust order statistic of within-bin differences | **No** |
| Shannon entropy | Entropy of the pair differences | **No** |
| Percentile | Selected percentile of the pair differences | **No** |
| Custom callable | User-defined calculation | **No** |

### Variography

{meth}`ds.rst.variogram() or Raster.variogram() <RasterBase.variogram>`<br>
{meth}`gdf.pc.variogram() or PointCloud.variogram() <PointCloudBase.variogram>`

An empirical variogram groups sampled pairs by distance and estimates their semivariance. A fitted model describes
this spatial variability with a correlation range, a structured variance (partial sill) and an optional nugget.

Install `geoutils[geostat]` for the optional geostatistical backends. GeoUtils uses SciKit-GStat estimators and models
and returns a {class}`~geoutils.Variogram` with the empirical bins and fitted parameters.

```{code-cell} ipython3
# Estimate elevation variability over distances up to 5 km using one pair sample
variogram = ds.rst.variogram(
    n_pairs=20_000,
    n_lags=12,
    max_lag=5000,
    model="spherical",
    random_state=42,
)
variogram.to_dataframe().head()
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

fig, ax = plt.subplots(figsize=(6, 3.5))
variogram.plot(ax=ax)
ax.set(xlabel="Distance (m)", ylabel="Elevation semivariance (m²)")
plt.tight_layout()
```

**Distance uses coordinate units; semivariance uses squared value units.** This example describes variability in
terrain elevation. To estimate an error variogram, use error measurements or elevation differences on stable terrain.

#### Sampling and fitting

**Logarithmic distance sampling represents short and long separations efficiently.** `n_pairs` sets the sample size
and `n_lags` sets the number of distance bins. The default uses one sample, with chunk scheduling handled by the
sampling backend. See {ref}`sampling-pairs` for pair selection.

For an advanced estimate of sampling variability, set `n_runs=3` or more to repeat sampling with shared distance
bins. The result includes the standard error across those independent estimates. This describes sampling
variability; it does not give confidence intervals on fitted parameters. Repetitions run sequentially, while each
sample retains its backend's chunk scheduling.

Fitting is optional. Use `.fit()` to fit or refit the empirical bins without repeating the sampling:

```{code-cell} ipython3
# Fit another model to the retained empirical variogram
refitted = variogram.fit("gaussian")
refitted.model.to_dict()
```

Use `model=["spherical", "gaussian"]` to fit a **sum of models at different spatial scales**.
{meth}`~geoutils.Variogram.from_pairs` estimates an empirical variogram from an existing pair sample. The result
stores bin statistics and model parameters, so the pair arrays can be discarded.

#### Reusing models

A {class}`~geoutils.Variogram` stores model parameters and evaluates **semivariance, covariance and correlation**.
Export it to a dataframe, Xarray dataset or dictionary to retain its bins, counts and fitted model:

```{code-cell} ipython3
# Keep a portable result and evaluate correlation at selected distances
dataset = variogram.to_xarray()
restored = gu.Variogram.from_dict(variogram.to_dict())
restored.correlation(np.array([0, 100, 1000]))
```

Use {meth}`~geoutils.Variogram.from_model` to define a model from known parameters. Convert it to GSTools or GPyTorch
to use the same spatial model in **kriging, random field simulation or Gaussian process calculations**:

```{code-block} python
known = gu.Variogram.from_model("gaussian", effective_range=500, partial_sill=4, nugget=0.2)
gstools_model = known.to_gstools(dim=2).model

# GPyTorch receives the nugget separately as observation noise
conversion = known.to_gpytorch()
kernel, noise = conversion.kernel, conversion.noise
```

(stats-chunked-strategies)=
## Chunked strategies

This section covers advanced controls for grouped statistics on chunked data. `strategy` changes how partial group
results are combined; it does not change group membership or the requested estimator.

| Strategy | How chunks are combined | Best suited to |
| --- | --- | --- |
| `"dense"` | Keep a summary slot for every declared group | Mergeable estimators with a moderate number of groups |
| `"sparse"` | Keep summaries only for groups present in each chunk | Mergeable estimators with many or sparsely populated groups |
| `"groupwise"` | Gather the observations belonging to complete groups | Median, NMAD, quantiles and custom estimators |

The default, `strategy="auto"`, uses `groupwise` for non-mergeable estimators. For mergeable estimators, it uses
`dense` for at most 4096 declared group combinations and `sparse` above that. A complete group must still fit in
memory for a non-mergeable estimator, and `observed=False` can produce a large in-memory result table.

Dask follows the input chunks and scheduler. Eager arrays and GeoUtils objects can instead use a
{class}`~geoutils.multiproc.MultiprocConfig` to distribute tiles across workers. The numerical reductions are shared.

Sampling is independent of the reduction strategy. `subsampling_strategy="topk"` keeps sampled locations stable
across chunk layouts for a fixed seed, while `"sequential"` follows traversal order. With grouped statistics,
`subsample_per_group=True` applies the requested sample limit independently to every combined group. See
{ref}`sampling-reproducibility` for details.
