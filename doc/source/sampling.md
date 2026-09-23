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

GeoUtils implements functionalities to **sample data from rasters and point clouds** efficiently on large datasets.

Three types of sampling operations are supported:

- {ref}`sampling-subsample` selects a **random sample of values** in a single dataset,
- {ref}`sampling-cosample` selects the **co-located sample of values** common to two primary datasets,
- {ref}`sampling-pairs` selects **pairs of values within a single dataset**.

For all of these, GeoUtils supports **selecting only valid values** to guarantee an exact usable sample size (which is not trivial on datasets not held in memory!).

**Co-sampling** is especially useful for analyses requiring multiple large datasets (such as image registration),
while **pair-sampling** is at the core of spatial statistics (variography).
**Subsampling** is used widely across methods, and is available within co-sampling and pair-sampling directly.

```{note}
Sampling can be directly controlled through statistical operations, see the **{ref}`stats` feature page**.
Use the methods below to use samples directly in your workflows.

All methods on this page accept combinations of **out-of-memory rasters, point clouds and vectors**. They normally
follow the chunks or partitions used when opening each input; see {ref}`sampling-reproducibility` for advanced controls.
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

# Open a projected elevation raster and glacier outlines for the sampling examples
ds = gu.open_raster(gu.examples.get_path("exploradores_aster_dem"))
glaciers = gu.open_vector(gu.examples.get_path("exploradores_rgi_outlines"))
```

(sampling-subsample)=
## Subsampling

{meth}`ds.rst.subsample() or Raster.subsample() <RasterBase.subsample>`<br>
{meth}`gdf.pc.subsample() or PointCloud.subsample() <PointCloudBase.subsample>`

Subsampling selects **a subset of locations without replacement**, by default including only valid values.

A subsample size between 0 and 1 selects a fraction of the data, a larger value sets the maximum number of samples.

By default, the result is a point cloud that keeps the values, coordinates and CRS: accessor calls return a {class}`geopandas.GeoDataFrame`, while GeoUtils objects return a
{class}`~geoutils.PointCloud`.

```{code-cell} ipython3
# Select 2000 valid raster cells as a point cloud
points = ds.rst.subsample(2000, random_state=42)
points.head()
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

fig, axes = plt.subplots(1, 2)
ds.rst.plot(ax=axes[0], cmap="terrain", vmin=300, vmax=4000, cbar_title="Elevation (m)")
axes[0].set_title("Complete raster")
points.pc.plot(
    ref=ds,
    ax=axes[1],
    cmap="terrain",
    vmin=300,
    vmax=4000,
    cbar_title="Elevation (m)",
    markersize=2,
)
axes[1].set_title("2000 sampled points")
axes[1].set_yticklabels([])
plt.tight_layout()
```

Use `mask` to limit eligible locations. Set `as_array=True` to return values alone, and add
`return_indices=True` to return source positions instead. Raster positions are returned as rows and columns; point
cloud positions refer to the original table.

```{code-cell} ipython3
# Select 10% of valid elevations inside glacier outlines as an array
glacier_values = ds.rst.subsample(0.1, mask=glaciers, as_array=True, random_state=42)
glacier_values[:5]
```

(sampling-cosample)=
## Co-sampling

{meth}`ds.rst.cosample() or Raster.cosample() <RasterBase.cosample>`<br>
{meth}`gdf.pc.cosample() or PointCloud.cosample() <PointCloudBase.cosample>`

Co-sampling selects **two datasets at common valid locations**, accounting for their spatial extents, nodata and
optional masks. Raster–point comparisons return a point cloud by default, keeping only point locations supported by
both inputs. Use `subsample` to select part of that common support.

```{code-cell} ipython3
# Create a random point cloud extending beyond every side of the raster
rng = np.random.default_rng(42)
raster_width = ds.rst.bbox.right - ds.rst.bbox.left
raster_height = ds.rst.bbox.top - ds.rst.bbox.bottom
x_points = rng.uniform(ds.rst.bbox.left - raster_width / 4, ds.rst.bbox.right + raster_width / 4, 2000)
y_points = rng.uniform(ds.rst.bbox.bottom - raster_height / 4, ds.rst.bbox.top + raster_height / 4, 2000)

# Give the points a smooth synthetic elevation field with small random variations
x_scaled = (x_points - ds.rst.bbox.left) / raster_width
y_scaled = (y_points - ds.rst.bbox.bottom) / raster_height
point_elevation = 1800 + 900 * y_scaled + 500 * np.sin(2 * np.pi * x_scaled) + rng.normal(0, 100, 2000)
random_points = gu.PointCloudAccessor.from_xyz(
    x_points,
    y_points,
    point_elevation,
    crs=ds.rst.crs,
    data_column="elevation",
)

# Keep finite values shared by the raster and points inside the glacier outlines
paired = ds.rst.cosample(random_points, mask=glaciers)
paired.head()
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

from matplotlib.lines import Line2D
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
ds.rst.plot(ax=axes[0], cmap="terrain", vmin=300, vmax=4000, cbar_title="Elevation (m)")
random_points.pc.plot(ax=axes[0], color="darkred", add_cbar=False, marker="x", markersize=3)
glaciers.vct.plot(ref=ds, ax=axes[0], ec="k", fc="none", linewidth=0.6)
axes[0].set_xlim(x_points.min(), x_points.max())
axes[0].set_ylim(y_points.min(), y_points.max())
axes[0].set_title("All input datasets")
axes[0].legend(
    handles=[
        Line2D([0], [0], color="k", label="Vector mask"),
        Line2D([0], [0], color="darkred", marker="x", linestyle="none", label="Point cloud"),
    ],
    loc="lower right",
    fontsize=8,
    markerscale=0.8,
)

paired.pc.plot(
    column="self",
    ref=ds,
    ax=axes[1],
    cmap="terrain",
    vmin=300,
    vmax=4000,
    cbar_title="Raster elevation (m)",
    marker="x",
    markersize=5,
)
glaciers.vct.plot(ref=ds, ax=axes[1], ec="k", fc="none", linewidth=0.6)
axes[1].set_title("Valid raster values\nco-located with points")

paired.pc.plot(
    column="other",
    ref=ds,
    ax=axes[2],
    cmap="terrain",
    vmin=300,
    vmax=4000,
    cbar_title="Point elevation (m)",
    marker="x",
    markersize=5,
)
glaciers.vct.plot(ref=ds, ax=axes[2], ec="k", fc="none", linewidth=0.6)
axes[2].set_title("Valid point values\nco-located with raster")

axes[1].set_yticklabels([])
axes[2].set_yticklabels([])
plt.tight_layout()
```

The point output is a {class}`geopandas.GeoDataFrame` with `"self"` and `"other"` columns on the common locations.
GeoUtils object calls return the corresponding {class}`~geoutils.PointCloud`. Co-sampling on a raster grid instead
returns a {class}`xarray.DataArray` or {class}`~geoutils.Raster`.

### Choosing the output locations

Use `at="self"`, `at="other"` or a spatial object to choose the exact output locations. Raster–point comparisons
use point locations by default. Their conversion direction can also be set explicitly:

| Mode | Output locations | Value calculation |
| --- | --- | --- |
| `"resample_raster"` | Point coordinates | Evaluate rasters with `resample_method`, passed to {meth}`interp_points() <RasterBase.interp_points>` |
| `"grid_points"` | Raster grid | Grid point values with `grid_method`, passed to {meth}`grid() <PointCloudBase.grid>` |

Both interpolation and gridding default to `"linear"`. Pass method-specific options through `resample_kwargs` or
`grid_kwargs`. For example, the points can be averaged onto the raster grid with:

```python
on_grid = ds.rst.cosample(
    random_points,
    raster_point_mode="grid_points",
    grid_method="mean",
    grid_kwargs={"dist_nodata_pixel": 2, "min_points": 3},
)
```

### Auxiliary variables

Use `auxiliary` to carry predictors, weights or other values through the same selection. Spatial inputs provide
their coordinates; use `auxiliary_at` to identify the primary grid or point order followed by a plain array.

```{code-cell} ipython3
# Carry the point identifiers through the same common selection
with_auxiliary = ds.rst.cosample(
    random_points,
    auxiliary={"point_id": np.arange(len(random_points))},
    auxiliary_at="other",
)
with_auxiliary.columns
```

Outputs store `"self"`, `"other"`, then auxiliary names in mapping order.

(sampling-pairs)=
## Pair-sampling

{meth}`ds.rst.pairsample() or Raster.pairsample() <RasterBase.pairsample>`<br>
{meth}`gdf.pc.pairsample() or PointCloud.pairsample() <PointCloudBase.pairsample>`

Pair-sampling selects **pairs of valid locations and their spatial separation** without calculating every possible
combination. The example samples raster cells across logarithmic distances, then selects four spatially separated
pairs for a closer look:

```{code-cell} ipython3
# Sample raster pairs over distances up to 5 km
n_pair_samples = 750
pairs = ds.rst.pairsample(
    n_pairs=n_pair_samples,
    max_distance=5000,
    random_state=42,
    strategy="independent",
)
pairs
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

from matplotlib.ticker import MaxNLocator

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
endpoint_colors = ["tab:blue", "tab:orange"]
for axis in axes:
    ds.rst.plot(ax=axis, cmap="gray", vmin=300, vmax=4000, add_cbar=False, alpha=0.3)
    axis.xaxis.set_major_locator(MaxNLocator(4))

# Plot every sampled endpoint with opaque, lightweight markers
endpoint_markers = ["o", "x"]
for endpoint, (color, marker) in enumerate(zip(endpoint_colors, endpoint_markers)):
    axes[0].scatter(
        pairs["x"].isel(endpoint=endpoint),
        pairs["y"].isel(endpoint=endpoint),
        s=5,
        color=color,
        marker=marker,
        linewidths=0.4,
        label=f"Endpoint {endpoint + 1}",
    )
axes[0].set_title("All sampled pair endpoints")
axes[0].legend(loc="lower right", fontsize=8, markerscale=2)

# Select different raster quadrants and representative distances for the detailed examples
bbox = ds.rst.bbox
midpoint_x = pairs["x"].mean("endpoint").values
midpoint_y = pairs["y"].mean("endpoint").values
distance = pairs["distance"].values
pair_targets = [(0.25, 0.75, 500), (0.75, 0.75, 1500), (0.25, 0.25, 3000), (0.75, 0.25, 4500)]
example_indices = []
for x_fraction, y_fraction, target_distance in pair_targets:
    target_x = bbox.left + x_fraction * (bbox.right - bbox.left)
    target_y = bbox.bottom + y_fraction * (bbox.top - bbox.bottom)
    spatial_score = ((midpoint_x - target_x) / (bbox.right - bbox.left)) ** 2
    spatial_score += ((midpoint_y - target_y) / (bbox.top - bbox.bottom)) ** 2
    distance_score = (np.log(distance) - np.log(target_distance)) ** 2
    score = 12 * spatial_score + distance_score
    score[example_indices] = np.inf
    example_indices.append(np.argmin(score))
example_pairs = pairs.isel(pair=example_indices)

for pair_number in range(example_pairs.sizes["pair"]):
    pair = example_pairs.isel(pair=pair_number)
    axes[1].plot(pair["x"], pair["y"], color="0.25", linewidth=1)
    midpoint = pair[["x", "y"]].mean("endpoint")
    axes[1].annotate(
        str(pair_number + 1),
        (midpoint["x"], midpoint["y"]),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
    )
for endpoint, color in enumerate(endpoint_colors):
    axes[1].scatter(
        example_pairs["x"].isel(endpoint=endpoint),
        example_pairs["y"].isel(endpoint=endpoint),
        s=20,
        color=color,
        label=f"Endpoint {endpoint + 1}",
    )
axes[1].set_title("Four pairs from the sample")
axes[1].set_yticklabels([])
axes[1].legend(loc="lower right", fontsize=8)
plt.tight_layout()
```

The output is an {class}`xarray.Dataset` indexed by `pair` and `endpoint`, with values, coordinates, source indexes
and distances. Distances use the CRS units. Increase `n_pairs` for analysis; masks or strict distance limits can
produce fewer pairs than requested.

### Sampling across distances

The default `sampling="loglag"` balances short and long separations. Use `sampling="random_xy"` to select endpoints
uniformly, and `min_distance`, `max_distance` or `mask` to limit eligible pairs. See {ref}`api-sampling` for the
available strategies.

```{code-cell} ipython3
# Compare the log-lag sample above with uniformly selected endpoints
random_pairs = ds.rst.pairsample(
    n_pairs=n_pair_samples,
    sampling="random_xy",
    max_distance=5000,
    random_state=42,
)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

fig, ax = plt.subplots(figsize=(6, 3))
distance_edges = np.geomspace(ds.rst.res[0], 5000, 13)
ax.hist(pairs["distance"], bins=distance_edges, histtype="step", label="Log-lag")
ax.hist(random_pairs["distance"], bins=distance_edges, histtype="step", label="Random XY")
ax.set(xscale="log", xlabel="Distance (m)", ylabel="Number of pairs")
ax.legend()
plt.tight_layout()
```

To estimate a variogram from these pairs, use {meth}`~geoutils.Variogram.from_pairs`. Alternatively,
{meth}`variogram() <RasterBase.variogram>` performs both steps, as described in {ref}`stats`.

(sampling-reproducibility)=
## Reproducibility and chunked execution

Set `random_state` to reproduce a selection. The default `strategy="topk"` for subsampling and co-sampling keeps the
same seeded raster locations across chunk layouts; `"sequential"` follows the order of valid values. Pair-sampling
uses its own strategies, some of which depend on chunk layout.

Subsampling and co-sampling preserve lazy Dask outputs where possible. `pairsample()` returns an in-memory
{class}`xarray.Dataset`. Use `mp_config` for multiprocessing on eager or unloaded data; it cannot be combined with
Dask input. See {ref}`scalability-logic` for details of the chunked algorithms.
