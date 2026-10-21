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
Sampling can be directly controlled in statistical operations, see **{ref}`stats`**. Use the methods below when relying on samples directly.
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
By default, the result is a point cloud.

For example, we subsample points in a raster:

```{code-cell} ipython3
# Select 2000 valid raster cells as a point cloud
points = ds.rst.subsample(2000, random_state=42)
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

(sampling-cosample)=
## Co-sampling

{meth}`ds.rst.cosample() or Raster.cosample() <RasterBase.cosample>`<br>
{meth}`gdf.pc.cosample() or PointCloud.cosample() <PointCloudBase.cosample>`

Co-sampling selects **two datasets at common valid locations**, accounting for their spatial extents, nodata and
optional masks.

The **support** on which the datasets are sampled can be specified with `at`.

```{important}
If inputs are not aligned in CRS, or differ in type, cosampling will automatically resample them at the common support, respecting
their chunks.
For a **raster-point comparison**, use `raster_point_mode` to specify if you want the point input to be gridded,
or the raster input to be resampled at points. By default, raster-raster comparisons return the raster from which the function is called, while raster–point return
the support of the point cloud.
```

Below, we cosample a point and a raster, with a vector **mask** to further reduce the locations, and a **subsample** to further
reduce the output.


```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the code for creating the example point cloud"
:  code_prompt_hide: "Hide the code for creating the example point cloud"

# Create a narrow diagonal swath extending past the raster's left and right edges
rng = np.random.default_rng(42)
raster_width = ds.rst.bbox.right - ds.rst.bbox.left
raster_height = ds.rst.bbox.top - ds.rst.bbox.bottom
x_scaled = np.linspace(-0.15, 1.15, 800)
y_scaled = 0.2 + 0.6 * x_scaled + rng.uniform(-0.04, 0.04, x_scaled.size)
x_points = ds.rst.bbox.left + x_scaled * raster_width
y_points = ds.rst.bbox.bottom + y_scaled * raster_height

# Give the points a smooth synthetic elevation field with small random variations
point_elevation = 1800 + 900 * y_scaled + 500 * np.sin(2 * np.pi * x_scaled) + rng.normal(0, 100, x_scaled.size)
swath_points = gu.pointcloud.from_xyz(
    x_points,
    y_points,
    point_elevation,
    crs=ds.rst.crs,
    data_name="elevation",
)
```

```{code-cell} ipython3
# Keep finite values shared by the raster and points inside the glacier outlines
paired = ds.rst.cosample(swath_points, mask=glaciers, subsample=100, random_state=42)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

from matplotlib.lines import Line2D
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
ds.rst.plot(ax=axes[0], cmap="terrain", vmin=300, vmax=4000, cbar_title="Elevation (m)")
swath_points.pc.plot(ax=axes[0], color="darkred", add_cbar=False, marker="x", markersize=3)
glaciers.vct.plot(ref=ds, ax=axes[0], ec="k", fc="none", linewidth=0.6)
axes[0].set_xlim(x_points.min(), x_points.max())
axes[0].set_ylim(ds.rst.bbox.bottom, ds.rst.bbox.top)
axes[0].set_title("All input datasets")
axes[0].legend(
    handles=[
        Line2D([0], [0], color="k", label="Vector mask"),
        Line2D([0], [0], color="darkred", marker="x", linestyle="none", label="Point swath"),
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

The output contains two "self" and "other" in its columns or band variables. In this case, the default point output is a
{class}`xarray.DataArray`:

```{code-cell} ipython3
:tags: [hide-input]

paired
```

(sampling-pairs)=
## Pair-sampling

{meth}`ds.rst.pairsample() or Raster.pairsample() <RasterBase.pairsample>`<br>
{meth}`gdf.pc.pairsample() or PointCloud.pairsample() <PointCloudBase.pairsample>`

Pair-sampling selects **pairs of valid locations and their spatial separation** efficiently on large data.

The example below samples pairs of raster values, with a closer look on 4 pairs and their separation distance:

```{code-cell} ipython3
# Sample raster pairs over distances up to 5 km
pairs = ds.rst.pairsample(
    n_pairs=750, max_distance=5000, random_state=42, strategy="independent"
)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

from matplotlib.ticker import MaxNLocator

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
endpoint_colors = ["tab:blue", "tab:orange"]
endpoint_names = ["A", "B"]
for axis in axes:
    ds.rst.plot(ax=axis, cmap="gray", vmin=300, vmax=4000, add_cbar=False, alpha=0.3)
    axis.xaxis.set_major_locator(MaxNLocator(4))

# Plot every sampled endpoint
endpoint_markers = ["o", "x"]
for endpoint, (color, marker) in enumerate(zip(endpoint_colors, endpoint_markers)):
    axes[0].scatter(
        pairs["x"].isel(endpoint=endpoint),
        pairs["y"].isel(endpoint=endpoint),
        s=5,
        color=color,
        marker=marker,
        linewidths=0.4,
        label=f"Endpoints {endpoint_names[endpoint]}",
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
        xytext=(4, 10),
        textcoords="offset points",
        fontsize=8,
    )
for endpoint, color in enumerate(endpoint_colors):
    axes[1].scatter(
        example_pairs["x"].isel(endpoint=endpoint),
        example_pairs["y"].isel(endpoint=endpoint),
        s=20,
        color=color,
        label=f"Endpoints {endpoint_names[endpoint]}",
    )
axes[1].set_title("Four pairs from the sample")
axes[1].set_yticklabels([])
axes[1].legend(loc="lower right", fontsize=8)
plt.tight_layout()
```

The output is an {class}`xarray.Dataset` indexed by `pair` and `endpoint`, with values, coordinates, source indexes
and distances. Distances use the CRS units.

```{code-cell} ipython3
:tags: [hide-input]

pairs
```

The **default `sampling="loglag"` balances separations by sampling uniformly with logarithmic distance**,
while `sampling="random_xy"` select endpoints at random X/Y coordinates that can severely undersample close or far pairs.
Use `min_distance`, `max_distance` or `mask` to limit the location or distances of sampled pairs.

```{code-cell} ipython3
# Compare the log-lag sample above with uniformly selected endpoints
random_pairs = ds.rst.pairsample(
    n_pairs=750,
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
ax.hist(pairs["distance"], bins=distance_edges, histtype="step", label="Log-lag (default)")
ax.hist(random_pairs["distance"], bins=distance_edges, histtype="step", label="Random XY")
ax.set(xscale="log", xlabel="Distance (m)", ylabel="Number of pairs")
ax.legend()
plt.tight_layout()
```

To estimate a variogram from these pairs, use {meth}`~geoutils.Variogram.from_pairs`. Alternatively,
{meth}`~RasterBase.variogram` performs both steps, as described in {ref}`stats`.

## Advanced use

### Subsampling options

Set `as_array=True` to return **values alone**, and use `mask` to limit eligible locations. Add
`return_indices=True` to return **source positions** instead: rows and columns for rasters, or row positions for
point clouds.

Below, we select 10% of valid elevations inside the glacier outlines. Repeating the call with the same seed and
`return_indices=True` locates those values in the original raster.

```{code-cell} ipython3
# Select elevations and their source positions inside glacier outlines
glacier_values = ds.rst.subsample(0.1, mask=glaciers, as_array=True, random_state=42)
glacier_rows, glacier_columns = ds.rst.subsample(
    0.1, mask=glaciers, as_array=True, return_indices=True, random_state=42
)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

# Map the sampled source positions inside glacier outlines
glacier_x, glacier_y = ds.rst.ij2xy(glacier_rows, glacier_columns)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
ds.rst.plot(ax=axes[0], cmap="gray", add_cbar=False, alpha=0.3)
glaciers.vct.plot(ref=ds, ax=axes[0], ec="k", fc="none", linewidth=0.6)
axes[0].scatter(glacier_x, glacier_y, s=0.1, color="darkred")
axes[0].set_title("10% of valid glacier cells")

# Show the distribution of sampled elevations
axes[1].hist(glacier_values, bins=20, color="darkred")
axes[1].set(title="Sampled elevations", xlabel="Elevation (m)", ylabel="Number of cells")
plt.tight_layout()
```

### Choosing cosampled support

Use `at="self"`, `at="other"` or a spatial object to **choose the output locations**. Raster–point comparisons
use point locations by default. Their conversion direction can also be set with `raster_point_mode`:

| Mode | Output locations | Value calculation |
| --- | --- | --- |
| `"resample_raster"` | Point coordinates | Evaluate rasters with `resample_method`, passed to {meth}`interp_at_points() <RasterBase.interp_at_points>` |
| `"grid_points"` | Raster grid | Grid point values with `grid_method`, passed to {meth}`grid() <PointCloudBase.grid>` |

Both interpolation and gridding default to `"linear"`. Pass method-specific options through `resample_kwargs` or
`grid_kwargs`.

Here, we average point elevations onto a 500 m raster grid, using a neighborhood radius of two output pixels.
The `at="self"` argument selects the grid of the raster on which we call cosample().

```{code-cell} ipython3
# Create a coarser raster and average the points onto its grid
reference_grid = ds.rst.reproject(res=500)
on_grid = reference_grid.rst.cosample(
    swath_points,
    at="self",
    grid_method="mean",
    grid_kwargs={"radius": 2, "min_points": 3},
)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

# Compare the original point elevations with their means on the raster grid
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
swath_points.pc.plot(
    ref=ds, ax=axes[0], cmap="terrain", vmin=300, vmax=4000,
    cbar_title="Elevation (m)", markersize=4,
)
axes[0].set_title("Point elevations")

# Plot the second band, which contains the averaged point elevations
on_grid.isel(band=1).rst.plot(
    ax=axes[1], cmap="terrain", vmin=300, vmax=4000, cbar_title="Elevation (m)",
    interpolation="nearest",
)
axes[1].set_title("Mean point elevation\non a 500 m grid")
axes[1].tick_params(labelleft=False)
plt.tight_layout()
```

### Auxiliary variables

Use `auxiliary` to **sample additional variables at the same locations**. These can be predictors, weights or
other values. Spatial inputs provide their coordinates; use `auxiliary_at` to identify the primary grid or point
order followed by a plain array.

For example, we carry each point's original row number through the common selection. The same colors identify
these points on the map and in the elevation comparison.

```{code-cell} ipython3
# Carry the point identifiers through the same common selection
point_ids = np.arange(swath_points.pc.point_count)
with_auxiliary = ds.rst.cosample(
    swath_points,
    auxiliary={"point_id": point_ids},
    auxiliary_at="other",
)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

# Color both the map and the elevation comparison by the original point identifiers
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
with_auxiliary.pc.plot(
    column="point_id", ref=ds, ax=axes[0], cmap="viridis", vmin=0, vmax=point_ids[-1],
    cbar_title="Original point row", markersize=4,
)
axes[0].set_title("Selected point identifiers")

# Compare raster and point elevations with the same identifier colors
axes[1].scatter(
    with_auxiliary, with_auxiliary["other"], c=with_auxiliary["point_id"],
    cmap="viridis", vmin=0, vmax=point_ids[-1], s=4,
)
axes[1].set(
    title="Co-located elevations", xlabel="Raster elevation (m)", ylabel="Point elevation (m)"
)
plt.tight_layout()
```

For a point DataArray, `"self"` contains the active values, while `"other"` and the auxiliaries are coordinates.
Raster outputs store these variables as bands; GeoDataFrame outputs store them as columns.

(sampling-reproducibility)=
### Reproducibility and chunked execution

In construction
