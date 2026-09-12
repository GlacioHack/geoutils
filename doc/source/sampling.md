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
(sampling)=
# Sampling

GeoUtils contains functionalities to **sample subsets of valid data from rasters and point clouds**, all supporting **chunked execution** to scale efficiently on large datasets.

Three types of sampling operations are supported:

- **Subsampling** selects a **random subset of valid values** in a single dataset,
- **Co-sampling** selects the **co-located sample of valid values** common to two primary datasets,
- **Pair-sampling** selects **pairs of values within a single dataset**.

Co-sampling is especially useful for analyses requiring multiple large datasets (such as image registration),
while pair-sampling is at the core of geostatistics (variography and kriging). Subsampling is widely used across applications,
starting within co-sampling and pair-sampling themselves.

```{note}
Sampling can be run directly through statistical operations in {ref}`stats` (e.g., for summary statistics or variogram estimation).
Use the API below for custom usage.
```

## Summary and quick use

| Operation | Selection                                     | Result |
| --- |-----------------------------------------------| --- |
| {meth}`~geoutils.Raster.subsample` | Valid observations in one dataset             | Values or indexes |
| {meth}`~geoutils.Raster.cosample` | Locations valid in both datasets              | Raster or point cloud on the selected support |
| {meth}`~geoutils.Raster.pairsample` | Pairs of locations within one dataset | {class}`xarray.Dataset` with paired values and distances |


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

# Open a projected elevation raster and glacier outlines for the sampling examples
rast = gu.Raster(gu.examples.get_path("exploradores_aster_dem"))
glaciers = gu.Vector(gu.examples.get_path("exploradores_rgi_outlines"))
```

(sampling-subsample)=
## Subsampling

{meth}`~geoutils.Raster.subsample` or {meth}`~geoutils.PointCloud.subsample`.

Subsampling selects a **random subset of valid values, without replacement**.

Use `subsample` to set the subsample size: a fraction between 0 and 1 selects that proportion of valid data, while a number above one sets the requested count.
For example, `subsample=0.1` selects 10%, `subsample=1` keeps all valid values, and `subsample=1000` selects 1000 values.

```{code-cell} ipython3
# Select a reproducible valid subsample of 2000 elevations
sample = rast.subsample(subsample=2000, random_state=42, strategy="topk")
sample[:5]
```

Pass `mask` to restrict eligible locations **before calculating the sample size**. Boolean arrays keep True values,
while vector outlines keep locations inside their geometries. Missing mask entries are excluded. For raster
sampling, mask rasters must share the source grid. For point sampling, point masks must follow the source points'
ordered coordinates and CRS; raster masks must share their CRS and are read with nearest interpolation.

```{code-cell} ipython3
# Select 10% of valid elevations inside glacier outlines
glacier_sample = rast.subsample(0.1, mask=glaciers, random_state=42)
glacier_sample[:5]
```

Use `return_indices=True` to get **sample locations instead of values**, for example to select the same cells in
several aligned arrays. Raster indexes are rows and columns; point cloud indexes are positions in the original table.
Masks do not change these index positions, and sampled values keep the source dtype.

```{code-cell} ipython3
# Recover the same raster cells with the same sample size, strategy and seed
rows, columns = rast.subsample(2000, return_indices=True, random_state=42, strategy="topk")
np.array_equal(rast.data[rows, columns], sample)
```

To keep coordinates and georeferencing together with the sample, use {meth}`~geoutils.Raster.to_pointcloud`:

```{code-cell} ipython3
# Keep sampled elevations together with their coordinates and CRS
points = rast.to_pointcloud(subsample=2000, random_state=42)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
rast.plot(ax=axes[0], cmap="terrain", vmin=300, vmax=4000, cbar_title="Elevation (m)")
axes[0].set_title("Complete raster")
points.plot(ax=axes[1], cmap="terrain", vmin=300, vmax=4000, cbar_title="Elevation (m)", markersize=2)
axes[1].set_title("2000 sampled locations")
axes[1].set_yticklabels([])
plt.tight_layout()
```

(sampling-cosample)=
## Co-sampling

{meth}`geoutils.Raster.cosample` or {meth}`geoutils.PointCloud.cosample`.

Co-sampling selects **values from two datasets at the same (co-located) valid locations**. It accounts for georeferencing, nodata
and optional masks, so the selected values can be compared directly. Use `subsample` to additionally select a common valid subset.

For example, compare a DEM with a coarser version outside glacier outlines. `align="reproject"` allows the coarse
raster to be resampled onto the calling raster's grid:

```{code-cell} ipython3
# Compare fine and coarse elevations on the common grid outside glacier outlines
coarse = rast.reproject(res=90)
paired = rast.cosample(
    coarse,
    mask=glaciers,
    mask_mode="outside",
    align="reproject",
)
fine_elevation, coarse_elevation = paired.split_bands()
paired
```

The output has **two bands with a common grid and mask**: `"self"` for the calling raster and `"other"` for the second
raster. Standard raster operations can then calculate their difference:

```{code-cell} ipython3
# Subtract the two bands on their common finite support
differences = fine_elevation - coarse_elevation
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

fig, ax = plt.subplots(figsize=(5, 3))
differences.plot(ax=ax, cmap="RdBu", vmin=-50, vmax=50, cbar_title="Fine − coarse elevation (m)")
glaciers.plot(ref_crs=rast, ax=ax, ec="k", fc="none", linewidth=0.5)

# Keep the view on the raster extent even though the outlines extend beyond it
ax.set_xlim(rast.bounds.left, rast.bounds.right)
ax.set_ylim(rast.bounds.bottom, rast.bounds.top)
ax.set_title("Common sample outside glaciers")
plt.tight_layout()
```

### Common locations and output

Use `at` to choose the **output grid or point locations**. It accepts `"self"`, `"other"` or a spatial object.
By default, two rasters use the calling raster's grid, while a raster–point comparison uses the point locations.
Use `raster_point_mode` to choose the direction of a raster–point comparison independently of the calling object:

| Mode | Output locations | Value calculation |
| --- | --- | --- |
| `"resample_raster"` | Point coordinates | Evaluate rasters with `resample_method`, passed to {meth}`~geoutils.Raster.interp_points` |
| `"grid_points"` | Raster grid | Grid point values with `grid_method`, passed to {meth}`~geoutils.PointCloud.grid` |

Both methods default to `"linear"`. `resample_method="nearest"` selects the nearest raster value;
`resample_method="linear"` interpolates neighboring raster values at the target point coordinates.
For point output, all point cloud inputs must share the chosen ordered coordinates.

```{code-cell} ipython3
# Interpolate the coarse raster at selected point observations
at_points = coarse.cosample(
    points,
    raster_point_mode="resample_raster",
    resample_method="linear",
    subsample=500,
    random_state=42,
)
at_points.ds[["self", "other", "geometry"]].head()
```

**The output follows the selected locations and calling interface:**

| Locations | GeoUtils object call | Accessor call |
| --- | --- | --- |
| Raster grid | {class}`~geoutils.Raster` | {class}`xarray.DataArray` |
| Point coordinates | {class}`~geoutils.PointCloud` | {class}`geopandas.GeoDataFrame` |

A raster keeps its grid, with unsampled cells masked. A point cloud contains only selected points and keeps their
original index labels and order. To compare two rasters on a point set, pass that point cloud as `at`.

Use one object family throughout a call: Raster/PointCloud objects, or DataArray/GeoDataFrame objects accessed
through `.rst` and `.pc`. This also applies to spatial auxiliaries, explicit `at`, and raster or point masks.
DataArrays and GeoDataFrames may mix eager and Dask storage. Plain arrays and vector outlines work with either
family. For point output, all point inputs must share the same ordered horizontal coordinates; this is checked
before raster alignment or interpolation. Point inputs may have different locations when gridded onto a raster.

An explicit `at` selects exact locations and determines the conversion direction when the mode is omitted.
When both are specified, they must agree. With an explicit mode and no `at`, exactly one primary input must supply
the requested spatial type; otherwise choose `at` explicitly. Omitting both retains the defaults described above.
Grid or CRS mismatches raise unless `align="reproject"` is set.

For example, use circular means of nearby point observations on a raster grid:

```python
on_grid = coarse.cosample(
    points,
    raster_point_mode="grid_points",
    grid_method="mean",
    grid_kwargs={"dist_nodata_pixel": 2, "min_points": 3},
)
```

Here the radius is two output pixels and each estimate requires three finite points. Gridding estimates values at
grid locations and can change the number and spatial distribution of observations used in a comparison.
Additional gridding options belong in `grid_kwargs`; raster interpolation options such as `nodata_propagation`
belong in `resample_kwargs`. Target locations and method names use the explicit co-sampling arguments.

`resample_method="reduce"` is reserved for reducing raster windows around point coordinates. It currently raises
`NotImplementedError` because its integration requires revision of {meth}`~geoutils.Raster.reduce_points`.
That existing method remains available separately. Co-sampling now uses `resample_method` in place of its previous
`interpolation` argument; the `stats` interpolation argument for grouped calculations is unchanged.

### Auxiliary variables

Auxiliary variables carry **additional values at the same common locations**, such as terrain attributes or
measurement weights. Locations must be valid in both primary datasets and every auxiliary variable.

Spatial objects supply their own georeferencing and use the first raster band or active point values by default.
Select another band with `auxiliary={"slope": (slope_raster, 2)}` or a point column with
`auxiliary={"intensity": (points, "intensity")}`. For plain arrays, `auxiliary_at` identifies the input whose grid
or point ordering they follow. Raster arrays must match one input band's shape; point arrays must be one-dimensional
with one value per input point:

```{code-cell} ipython3
# Carry an aligned elevation predictor through the same selection
with_auxiliary = rast.cosample(
    coarse,
    auxiliary={"elevation": rast.data},
    auxiliary_at="self",
    align="reproject",
)
first, second, elevation = with_auxiliary.split_bands()
with_auxiliary.tags["long_name"]
```

Bands or columns contain `"self"`, `"other"`, then auxiliaries in mapping order. Raster band names are stored in
`tags["long_name"]`, or `attrs["long_name"]` for Xarray. Point clouds use these names as columns, with `"self"`
as the active data column. Use the usual `.data`, `.ds` or `.split_bands()` methods to access the values.

```{code-cell} ipython3
# Use native Xarray bands for the same grid comparison
native = coarse.to_xarray().rst.cosample(coarse.to_xarray())
native.isel(band=0) - native.isel(band=1)
```

(sampling-pairs)=
## Pair-sampling

{meth}`geoutils.Raster.pairsample` or {meth}`geoutils.PointCloud.pairsample`.

Pair-sampling selects **pairs of valid values and their spatial separation within one dataset**. It provides the
observations used in variography without calculating distances between every possible pair of locations.

```{code-cell} ipython3
# Sample pairs at short and long separations within 5 km
pairs = rast.pairsample(n_pairs=20_000, max_distance=5000, random_state=42)
pairs.isel(pair=slice(0, 5))
```

The output is an {class}`xarray.Dataset` indexed by `pair` and `endpoint`. It contains both values, their coordinates
and source indexes, and their distance. **Distances use the coordinate units**: use a projected CRS in metres for
distances in metres.

```{code-cell} ipython3
# Calculate signed differences while keeping the pair labels
pair_differences = pairs["value"].sel(endpoint="second") - pairs["value"].sel(endpoint="first")
pair_differences.isel(pair=slice(0, 5))
```

### Sampling across distances

**Logarithmic distance sampling** (`sampling="loglag"`, the default) represents both short and long separations.
**Uniform endpoint sampling** (`sampling="random_xy"`) selects locations uniformly, which usually gives fewer pairs
at short distances. Use `min_distance`, `max_distance` and `mask` to restrict the selection.

```{code-cell} ipython3
# Compare logarithmic distance sampling with uniformly selected endpoints
random_pairs = rast.pairsample(
    n_pairs=20_000, sampling="random_xy", max_distance=5000, random_state=42
)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

fig, ax = plt.subplots(figsize=(6, 3))
distance_edges = np.geomspace(rast.res[0], 5000, 13)
ax.hist(pairs["distance"], bins=distance_edges, histtype="step", label="Logarithmic distances")
ax.hist(random_pairs["distance"], bins=distance_edges, histtype="step", label="Uniform endpoints")
ax.set(xscale="log", xlabel="Distance (m)", ylabel="Number of pairs")
ax.legend()
plt.tight_layout()
```

`n_pairs` sets a target count. Sparse data, masks or distance limits can leave fewer accepted pairs;
`pairs.sizes["pair"]` gives the returned count. See {ref}`api-sampling` for the available sampling strategies.

To estimate a variogram from these pairs, use {meth}`~geoutils.Variogram.from_pairs`. Alternatively,
{meth}`~geoutils.Raster.variogram` or {meth}`~geoutils.PointCloud.variogram` performs sampling and estimation in one
call, as described in {ref}`stats-variograms`.

(sampling-reproducibility)=
## Reproducibility and chunked execution

**Set `random_state` to reproduce a sample** with the same inputs and sampling settings. Raster subsampling offers
two strategies:

- `"sequential"` draws from the sequence of valid values. It is the default for `subsample()` and can depend on chunk layout.
- `"topk"` selects the same cells regardless of chunk layout. It is the default for raster `cosample()` and grouped statistics.

For grouped `stats()`, pass this choice as **`subsampling_strategy`**. Its separate `strategy` argument controls
aggregation across chunks; see {ref}`stats-grouped`.

This guarantee applies to the `topk` raster sampler. Pair-sampling has its own strategies; for example,
`"chunk_anchors"` uses the chunk layout to select pairs.

**Co-sampling supports Dask and multiprocessing through its spatial operations.** Dask accessor calls keep raster
bands or point partitions lazy. Counts and sample positions are computed during preparation, including a check for
an empty common selection. Final point interpolation and removal of missing values stay lazy; computing the result
can therefore leave fewer points, or none, when interpolation spreads nodata into the selected locations. Raster
output keeps the complete grid, even when `subsample` limits the selected cells; point output preserves the selected
row order and index labels.

Pass `mp_config` directly to `cosample()` to use multiprocessing with eager or unloaded inputs. Raster output is
written by tiles to `mp_config.outfile`, and temporary intermediate files are cleaned automatically. Point output is
collected after interpolating the selected rows. Dask inputs and `mp_config` cannot be combined.

Pair-sampling returns an in-memory pair dataset and, for point clouds, currently reads all source coordinates and
values. Use an absolute sample count to limit the returned rows or pairs. See {ref}`scalability-logic` for the chunked
algorithms.
