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

GeoUtils provides **summary statistics, grouped statistics and variography for rasters and point clouds**, accounting
for **georeferencing and nodata**. The same methods are available on GeoUtils objects and their Xarray or GeoPandas
accessors.

Three types of statistical operations are supported:

- **Summary statistics** describe the **distribution of valid values** in a dataset,
- **Grouped statistics** describe values **within bins or categories** defined by other variables,
- **Variography** describes **spatial variability as a function of distance**.

**Zonal statistics are a special case of grouped statistics: vector features define the bins.** For example,
`stats()` can calculate the mean elevation of each glacier or catchment using its outline.

```{note}
Statistical operations can select samples directly, for example when estimating a variogram or grouped statistics.
See {ref}`sampling` to select observations for other analyses, and {ref}`api-statistics` for the API reference.
```

## Summary and quick use

| Operation | Calculation | Result |
| --- | --- | --- |
| {meth}`~geoutils.Raster.stats` | Statistics of all valid values or by bins, categories, or vector zones | Number, dictionary, or {class}`pandas.DataFrame` |
| {meth}`~geoutils.Raster.variogram` | Spatial variability across distances | {class}`~geoutils.Variogram` |

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
rast = gu.Raster(gu.examples.get_path("exploradores_aster_dem"))
glaciers = gu.Vector(gu.examples.get_path("exploradores_rgi_outlines"))
```

(stats-estimators)=
## Summary statistics

{meth}`geoutils.Raster.stats` or {meth}`geoutils.PointCloud.stats`.

Summary statistics describe **central values, spread and valid counts**. For rasters, they use the selected band;
for point clouds, they use the active {attr}`~geoutils.PointCloud.data_column` or the geometry's Z coordinate.
Built-in estimators exclude nodata.

```{code-cell} ipython3
# Compute the default summary statistics
rast.stats()
```

Request **one statistic for a number**, or **several statistics for a dictionary**. Use `"all"` to request every
available estimator and count.

```{code-cell} ipython3
rast.stats("mean")
```

```{code-cell} ipython3
rast.stats(["mean", "median", "std", "nmad"])
```

```{dropdown} Available estimators and counts
| Statistic | Meaning |
| --- | --- |
| Mean, median | Arithmetic average or middle value |
| Min, max | Smallest or largest value |
| Sum, sum of squares | Sum of values or of their squares |
| 90th percentile | Value below which 90% of observations fall |
| IQR | Difference between the 75th and 25th percentiles |
| LE90 | Difference between the 95th and 5th percentiles |
| NMAD | Median absolute deviation from the median, scaled by 1.4826 |
| RMSE | Square root of the mean squared value, useful for error data |
| Standard deviation | Spread around the mean |
| Valid count, total count | Number of finite observations and total number of locations |
| Percentage valid points | Valid count divided by total count, as a percentage |

With an inlier mask, additional counts describe the selected locations, their finite values and their proportions
of the mask and dataset.
```

**Robust estimators**, such as the median and NMAD, are less sensitive to outliers than the mean and standard deviation.
The array functions {func}`~geoutils.stats.nmad`, {func}`~geoutils.stats.linear_error` and {func}`~geoutils.stats.rmse`
are also available directly.

Custom estimators receive the data array and must handle its mask or NaNs. For example, count valid elevations
above a threshold:

```{code-cell} ipython3
def count_high_elevations(data: np.ndarray) -> int:
    """Count observations above 1500 m in the selected elevation data."""

    # Exclude masked and non-finite observations before counting high elevations
    values = np.ma.asarray(data).compressed()
    return int(np.count_nonzero(np.isfinite(values) & (values > 1500)))

rast.stats(count_high_elevations)
```

Use `mask` to **restrict the statistics to selected locations**:

```{code-cell} ipython3
# Summarize elevations inside all glacier outlines together
glacier_mask = glaciers.create_mask(rast)
rast.stats(["mean", "std", "valid count"], mask=glacier_mask)
```

To calculate a separate statistic for each glacier, use {ref}`zonal statistics<stats-zonal>` below.

(stats-grouped)=
## Grouped statistics

{meth}`geoutils.Raster.stats`, {meth}`geoutils.PointCloud.stats` or {func}`geoutils.stats.stats` with ``by``.

Grouped statistics describe **values within bins or categories of one or more variables**. Bins can be continuous
intervals, such as elevation bands, or discrete categories, such as land-cover classes or vector zones.

Use `by` to name the grouping variables and `values` to select the data to summarize. Raster values use band numbers
starting at one; point cloud values use column names. A mapping names those values in the output and can also contain
**external rasters, point clouds or `(object, band_or_column)` selections**. Vector attributes can provide numeric
values or grouping variables; supply `bins` to treat an attribute as continuous.

### Continuous bins

Use `bins` to define **intervals of a continuous variable**. For example, summarize elevations within elevation bands:

```{code-cell} ipython3
# Divide the first raster band into explicit elevation intervals
elevation_edges = [0, 1000, 2000, 3000, 4000]
elevation_stats, elevation_masks = rast.stats(
    ("mean", "min", "max"),
    by={"elevation": 1},
    values={"elevation": 1},
    bins={"elevation": elevation_edges},
    return_masks=True,
)
elevation_stats
```

The output table has **one row per group and one column per value–statistic combination**. Each value also has a
finite `count`. With `return_masks=True`, the same row key retrieves the group's geographic mask:

```{code-cell} ipython3
# Retrieve the same elevation interval in the table and on the raster grid
first_interval = elevation_stats.index[0]
first_mask = elevation_masks[first_interval]
elevation_stats.loc[first_interval]
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
rast.plot(ax=axes[0], cmap="terrain", cbar_title="Elevation (m)")
axes[0].set_title("Elevation raster")
first_mask.plot(ax=axes[1], cbar_title="Inside interval (1=yes, 0=no)")
axes[1].set_title(f"Elevation interval {first_interval} m")
axes[1].set_yticklabels([])
plt.tight_layout()
```

An integer, such as `bins={"elevation": 10}`, creates ten equal-width bins. Numeric edges include each lower edge
and the final upper edge. Pass a {class}`pandas.IntervalIndex` to choose which edges are included:

```{code-cell} ipython3
# Assign boundary elevations to the interval ending at that elevation
right_closed = pd.IntervalIndex.from_breaks(elevation_edges, closed="right")
rast.stats(by={"elevation": 1}, bins={"elevation": right_closed})
```

### Discrete categories

Use `categories` to define **discrete classes and their order**. A boolean glacier mask, for example, separates terrain
inside and outside the outlines:

```{code-cell} ipython3
# Compare elevation distributions inside and outside glaciers
glacier_stats = rast.stats(
    ("mean", "min", "max", "nmad"),
    by={"glacier": glacier_mask},
    values={"elevation": 1},
    categories={"glacier": [False, True]},
)
glacier_stats
```

The same operation applies to land-cover codes or other categorical variables. Values outside the declared categories
are excluded. Boolean and Pandas categorical inputs can supply their categories without an explicit declaration.

(stats-zonal)=
### Zonal statistics

**Zonal statistics use vector features as bins.** Pass `(vector, "column")` in `by` to group raster cells or points
by a vector attribute. **A unique feature ID gives one group per feature**; repeated IDs combine features into a
group. Categories are inferred from the column, so no `bins` or `categories` argument is needed.

For example, the glacier inventory's `RGIId` column identifies each glacier:

```{code-cell} ipython3
# Calculate separate elevation statistics for each glacier outline
glacier_zonal_stats = rast.stats(
    ("mean", "std", "min", "max"),
    by={"glacier": (glaciers, "RGIId")},
    values={"elevation": 1},
    observed=False,
)
glacier_zonal_stats.sort_values(("elevation", "count"), ascending=False).head()
```

Locations outside all zones are excluded, and each statistic uses the finite values in its zone. `observed=False`
retains zones without observations, with zero counts and undefined statistics. Use `return_masks=True` to retrieve
zone masks by the same feature IDs.

The same grouping applies to point measurements:

```{code-cell} ipython3
# Summarize a sample of point elevations within each glacier
points = rast.to_pointcloud(data_column_name="elevation", subsample=2000, random_state=42)
points.stats(("mean", "std"), by={"glacier": (glaciers, "RGIId")}).head()
```

Raster cells are assigned by their centres, and points by their coordinates. Grouping assigns each location to one
category; use non-overlapping zones for independent feature statistics. Vector and GeoDataFrame inputs are both
accepted, including when the raster values are backed by Dask.

```{note}
`by={"zone": (zones, "id")}` groups by feature IDs. A bare vector, `by={"inside": zones}`, groups the union of all
features into inside/outside categories. `mask=zones` instead restricts all groups to locations inside the features.
```

### Several variables and plotting

**Several grouping variables define their joint bins or categories**. For example, combine elevation bands with
glacier membership to compare elevation distributions inside and outside glaciers at similar elevations:

```{code-cell} ipython3
# Retain empty combinations so the complete comparison grid remains visible
joint_stats = rast.stats(
    ("median", "nmad"),
    by={"elevation": 1, "glacier": glacier_mask},
    values={"elevation": 1},
    bins={"elevation": elevation_edges},
    categories={"glacier": [False, True]},
    observed=False,
)
joint_stats
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

# Display within-group elevation spread and the number of observations supporting it
fig, ax = plt.subplots(figsize=(7, 4))
fig.subplots_adjust(right=0.78)
axes = gu.stats.plot_grouped_stats(joint_stats, value="elevation", statistic="nmad", min_count=100, ax=ax)
axes["statistic"].set(xlabel="Elevation (m)", ylabel="Inside glacier")

# Place the color scale beyond the count panels so every elevation interval stays visible
axes["colorbar"].set_position((0.84, 0.11, 0.025, 0.5))
_ = axes["colorbar"].set_ylabel("Elevation NMAD (m)")
```

To summarize only one variable, make a separate grouping call. Statistics such as medians cannot generally be
combined from subgroup results.

### Choosing a grouped reduction strategy

`strategy` controls **how values belonging to a group are combined across chunks**. The names `dense` and `sparse`
describe intermediate group summaries. They do not describe the number of finite input pixels, select different
groups, or change the requested statistic. `observed` independently controls whether empty groups appear in the result.

| Strategy | What each chunk contributes | Suitable calculations and limits |
| --- | --- | --- |
| `"dense"` | A summary slot for every declared group combination, including groups absent from the chunk | Counts, means, standard deviations, sums, RMSE and extrema. Avoids matching group IDs during merges, but allocation grows with the complete group space |
| `"sparse"` | Summaries and IDs for groups encountered in that chunk | The same mergeable statistics. Useful when each chunk contains a small fraction of the declared groups; sorting and merging IDs adds work |
| `"groupwise"` | The actual observations for complete groups | Exact medians, quantiles, NMAD and custom functions. Reads intersecting chunks and batches small groups sharing those chunks; a large group still requires memory proportional to its full membership |
| `"auto"` | Chooses one of the above | Uses groupwise for exact quantiles, NMAD or custom functions; otherwise dense for at most 4096 declared group combinations, sparse above that |

For example, 100 elevation bins combined with 100 land-cover categories declare **10,000 combinations**, even if
only a few occur. Dense keeps slots for all combinations in each chunk. Sparse initially stores only those present;
its summaries can grow as chunks are merged. This distinction is most useful when groups occupy localized regions.
If every chunk contains every group, sparse has little allocation advantage.

The automatic threshold is a heuristic based on declared group count. It does not measure group occupancy or
available memory. Requesting an exact median alongside a mean selects groupwise for the entire calculation.
For chunked input, requesting those exact statistics with dense or sparse raises an error. Eager input can compute
exact statistics directly because its complete values are already resident in memory.

Dask and multiprocessing share the numerical kernels. Dask can read arrays in chunks; multiprocessing currently
tiles arrays already available in the client. Neither the choice of strategy nor smaller chunks bounds the memory
needed by one very large exact group. The result table is returned in memory, and `observed=False` can itself produce
a large table of every declared group combination.

Groupwise execution may reread a chunk when several batches need it. Smaller chunks can reduce temporary array
sizes while increasing the number of gathering tasks, so they do not necessarily make exact statistics faster.

`subsampling_strategy` is a separate control: `"topk"` keeps the selected raster cells independent of chunk layout
for a fixed seed, while `"sequential"` follows the ordinary sampling workflow. Subsampling changes which observations
enter the estimates; the reduction strategy controls how those selected observations are combined. Small floating
point differences can arise from different summation orders across chunk layouts.

Set **`subsample_per_group=True`** to sample within each group. For example, this uses at most 1,000 eligible
locations per land-cover category:

```python
rast.stats(
    ["mean", "std", "nmad"],
    by={"landcover": landcover},
    categories={"landcover": classes},
    subsample=1000,
    subsample_per_group=True,
    random_state=42,
)
```

With multiple grouping variables, the limit applies to each combined group. A fraction such as `subsample=0.1`
keeps ten percent of each group's eligible locations, rounded down; very small groups may receive no sample.
`subsample=1` keeps all eligible locations. Smaller groups keep all their locations when the requested maximum
exceeds their size. Every selected value column uses the same sampled locations, so missing values can lower its
finite count. Returned masks and observed group rows still describe membership before sampling. Without `by`,
the option uses the ordinary global sample.

The ASV grouped-statistics comparisons vary raster size, chunk size, group count and local versus interleaved
membership. They report moments separately from exact median/NMAD calculations, with complete result computation
inside the measured operation. Larger-than-memory tests additionally check Dask worker memory and health.

### Masks, alignment and chunked execution

**Group masks retain all eligible locations in each group**, including locations with missing values in a selected
band or column. Subsampling affects the statistics, not these masks. Masks retain their spatial representation and
can be plotted or saved, for example with `first_mask.to_file("elevation_bin.tif")`.

**Spatial preparation is shared with co-sampling.** Combining rasters uses the calling grid; combining rasters and
points uses the first point dataset's locations. Set `at` explicitly to choose another support. Raster values are
interpolated at points, and `align="reproject"` allows mismatched grids or CRSs to be aligned. Separate point datasets
must share the same ordered locations. Each selected value keeps its own finite count after alignment.

For arrays already aligned to one another, use the array function directly:

```{code-cell} ipython3
# Apply the same elevation grouping without geospatial objects
gu.stats.stats(
    {"elevation": rast.data},
    ("mean", "min", "max"),
    by={"elevation": rast.data},
    bins={"elevation": elevation_edges},
)
```

**Dask and multiprocessing use the same aggregation kernels** and return an in-memory table. Dask follows the input
chunks and scheduler. For eager arrays or GeoUtils objects, pass a {class}`~geoutils.multiproc.MultiprocConfig` configuration
from `geoutils.multiproc` to distribute array tiles across workers. Spatial preparation precedes aggregation.

Use `strategy` to control how groups are combined across chunks:

| Strategy | How it computes statistics | Useful for |
| --- | --- | --- |
| `"dense"` | Reduce each chunk into an accumulator for every declared group, then combine summaries | Moderate numbers of bins or categories |
| `"sparse"` | Reduce and combine only group IDs present in each chunk | Many zones or sparsely populated group combinations |
| `"groupwise"` | Gather complete groups from intersecting chunks, batching small groups that share chunks | Exact medians, NMAD, quantiles and custom functions |

**`strategy="auto"` chooses `groupwise` for exact statistics**, including median and NMAD in the default set. For
mergeable statistics such as count, mean, standard deviation or sum, it uses `dense` up to 4096 declared group
combinations and `sparse` above that. These defaults favor fast dense reductions while limiting the size of
intermediate summaries.
The `sparse` strategy uses ordinary NumPy arrays of observed groups; no sparse array dependency is required.

Exact statistics require a complete group's values to fit in memory. Use `subsample` and `random_state` for a
reproducible sampled estimate. **`subsampling_strategy` controls sample selection separately from aggregation**;
see {ref}`sampling-reproducibility` for its `"topk"` and `"sequential"` options.

(stats-variograms)=
## Variography

Use {func}`geoutils.stats.variogram`, {meth}`geoutils.Raster.variogram`, or
{meth}`geoutils.PointCloud.variogram`.

Variography describes **how differences between values change with spatial separation**. An empirical variogram
groups sampled pairs by distance and estimates their semivariance. A fitted model describes this spatial variability
with a correlation range, a structured variance (partial sill) and an optional nugget.

Install `geoutils[geostat]` for the optional geostatistical backends. GeoUtils uses SciKit-GStat estimators and models
and returns a {class}`~geoutils.Variogram` with the empirical bins and fitted parameters.

```{code-cell} ipython3
# Estimate elevation variability over distances up to 5 km using one pair sample
variogram = rast.variogram(
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

fig, ax = plt.subplots(figsize=(6, 3))
variogram.plot(ax=ax)
ax.set(xlabel="Distance (m)", ylabel="Elevation semivariance (m²)")
plt.tight_layout()
```

**Distance uses coordinate units; semivariance uses squared value units.** This example describes variability in
terrain elevation. To estimate an error variogram, use error measurements or elevation differences on stable terrain.

### Sampling and fitting

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

### Reusing models

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
