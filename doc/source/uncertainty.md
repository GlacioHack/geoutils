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
mystnb:
  execution_timeout: 120
---
(uncertainty)=
# Uncertainty

GeoUtils implements tools to **estimate and propagate uncertainty** for spatial calculations.

We focus on errors whose **magnitudes vary spatially** and/or that are **spatially correlated** for rasters
or point clouds, with methods to both estimate and propagate these efficiently on large datasets. This is done both
by using chunked execution in underlying operations, and by avoiding storing a full error covariance matrix.

Uncertainty quantification in GeoUtils strives to be **composable**, for ease of use.
For estimating error structure, we build upon grouped statistics and variography outputs from {ref}`stats`.
Then, an error structure can be used as input either for **weights** or to **propagate uncertainty** through most of GeoUtils
operations and beyond, either numerically or analytically.

Below, we first demonstrate how to {ref}`uncertainty-estimation`, then how to {ref}`uncertainty-propagation`, followed by more
advanced examples.

```{seealso}
GeoUtils does not handle errors correlated between different variables, such as multiple bands of the same instruments.
For these, see packages such as [ObsArray](https://obsarray.readthedocs.io/en/latest/)!
```

```{caution}
The uncertainty API is still **experimental**. It may change in future releases.
```

```{code-cell} ipython3
:tags: [remove-cell]

# Match the figure resolution and text size used by the other feature pages
from matplotlib import pyplot as plt

plt.rcParams["figure.dpi"] = 600
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["font.size"] = 9
```

(uncertainty-quick-use)=
## Quick use

To estimate an **error structure** (error components of different magnitude and spatial correlation), we rely on an **error proxy**,
typically the difference to another estimate (measurement or model).

In the synthetic example below, we compare a snow depth product to **a precise reference** (e.g. from lidar, could also be ground measurements),
for which we assume negligible errors.

We then estimate the error structure using **viewing angle** and **land cover** as predictors of error variability.

```{tip}
Finding an error proxy or predictors can be application-specific, but often comes down to similar techniques.
See {ref}`uncertainty-error-proxy` and {ref}`uncertainty-error-predictors` for hints!
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for creating synthetic snow depth"
:  code_prompt_hide: "Hide the code for creating synthetic snow depth"

import numpy as np
from affine import Affine
from scipy.ndimage import gaussian_filter

import geoutils as gu

# Create snow depth on a 10 m grid
quick_size = 96
quick_extent = quick_size * 10
quick_rows, quick_columns = np.indices((quick_size, quick_size))
cover_codes = [0, 1, 2]
cover_names = ["Forest", "Meadow", "Shrubs"]

# Make irregular land cover areas by smoothing random noise
quick_random = np.random.default_rng(44)
cover_field = gaussian_filter(quick_random.normal(size=quick_rows.shape), sigma=6)
cover_thresholds = np.quantile(cover_field, [0.5, 0.8])
quick_cover = np.digitize(cover_field, cover_thresholds)

# Create snow depth and viewing angles
quick_values = (
    np.take([1.5, 2.4, 2.35], quick_cover)
    + 0.2 * np.exp(-((quick_columns - 30) ** 2 + (quick_rows - 52) ** 2) / (2 * 24**2))
    + 0.1 * np.cos(quick_rows / 14)
)
quick_transform = Affine(10, 0, 0, 0, -10, quick_extent)
quick_angles = 60 * (quick_columns + 0.5 * quick_rows) / (1.5 * (quick_size - 1))
ds_ref = gu.raster.from_array(quick_values, quick_transform, 32631)

# Create short and long range error patterns
quick_short_model = gu.Variogram.from_model("gaussian", effective_range=25, partial_sill=1)
quick_long_model = gu.Variogram.from_model("gaussian", effective_range=220, partial_sill=1)
quick_short_structure = gu.ErrorStructure([gu.ErrorComponent("short_range", 1.0, quick_short_model)])
# Make long range errors smaller to show land cover differences
quick_long_structure = gu.ErrorStructure([gu.ErrorComponent("long_range", 0.08, quick_long_model)])
quick_noise = ds_ref.rst.random_field(quick_short_structure, backend="gstools", random_state=42).values
quick_long_noise = ds_ref.rst.random_field(quick_long_structure, backend="gstools", random_state=43).values

# Let error size depend on land cover and viewing angle
quick_scales = np.take([1.5, 0.8, 1.15], quick_cover)
quick_magnitude = quick_scales * (0.15 + 0.004 * quick_angles)
quick_errors = quick_magnitude * quick_noise + quick_long_noise
ds = gu.raster.from_array(quick_values + quick_errors, quick_transform, 32631).assign_attrs(units="m")
view_angle = gu.raster.from_array(quick_angles, quick_transform, 32631).assign_attrs(
    long_name="Viewing angle", units="degrees",
)
land_cover = gu.raster.from_array(quick_cover, quick_transform, 32631).assign_attrs(
    long_name="Land cover", flag_values=cover_codes, flag_meanings="forest meadow shrubs",
)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the inputs"
:  code_prompt_hide: "Hide the code for plotting the inputs"
:  image:
:    width: 800px
:    alt: "Snow depth differences from a precise reference, viewing angle, and forest, meadow and shrub areas controlling error magnitude."

from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

cover_colors = ["forestgreen", "khaki", "tan"]

# Show the error proxy and its predictors
quick_proxy = ds - ds_ref
proxy_limit = np.nanmax(np.abs(quick_proxy.values))
figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
quick_proxy.rst.plot(
    ax=axes[0], cmap="RdBu", vmin=-proxy_limit, vmax=proxy_limit,
    cbar_title="Snow depth difference (m)",
    title="Error proxy\n(measurement - reference)",
)
view_angle.rst.plot(
    ax=axes[1], cmap="viridis", cbar_title="Viewing angle (°)",
    title="Predictor 1\n(viewing angle)",
)
land_cover.rst.plot(
    ax=axes[2], cmap=ListedColormap(cover_colors),
    vmin=-0.5, vmax=2.5, add_cbar=False, title="Predictor 2\n(land cover)",
)
legend_handles = [
    Patch(facecolor=color, label=label)
    for color, label in zip(cover_colors, cover_names)
]
axes[2].legend(handles=legend_handles, loc="upper right", framealpha=0.9)
for axis in axes:
    axis.set(xlabel="X (m)", xlim=(0, quick_extent), ylim=(0, quick_extent))
axes[0].set_ylabel("Y (m)")
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
:mystnb:
:  image:
:    width: 800px
:    alt: "Error magnitudes grouped by viewing angle and land cover, with the fitted spatial autocorrelation."

import geoutils as gu

# Estimate and plot the error structure
predictors = {"view_angle": view_angle, "land_cover": land_cover}
es = gu.ErrorStructure.estimate(
    ds,
    other=ds_ref,
    predictors=predictors,
    bins={"view_angle": 6},
    categories={"land_cover": cover_codes},
    random_state=42,
)
es.plot()
```
Note that, by default, variable errors are interpolated linearly along continuous variables (like viewing angle), and spatial correlation keeps the best fit after comparing a sum
of multiple variogram models.

Then, we can use this error structure to **propagate uncertainty** through any operation, such as {meth}`stats() <RasterBase.stats>` to calculate **mean snow depth per land-cover area**:

```{code-cell} ipython3
_, region_out = ds.rst.stats(
    "mean",
    by={"land_cover": land_cover},
    categories={"land_cover": cover_codes},
    error_structure=es,
    uncertainty_kwargs={"predictors": predictors},
)
```

And compare means in various land cover accounting for their uncertainties:

```{code-cell} ipython3
intervals = region_out.interval(0.95).to_numpy()
for cover, mean, (lower, upper) in zip(cover_names, region_out.estimate, intervals):
    print(f"{cover}: {mean:.2f} m (95% interval: {lower:.2f}–{upper:.2f} m)")
```

## Concept definitions

We briefly summarize **important concepts** for uncertainty quantification directly below. Skip to {ref}`uncertainty-estimation` to dive
directly into functionality examples.

(uncertainty-error-structure)=
### Error and uncertainty

An **error** is the difference between an **estimate** (measured or modelled) and its true, generally unknown value.
Errors can follow many distributions. For most applications, it is often useful to separate a
**systematic component** (bias) from a **random component** (fluctuations around zero).
Random errors are often **spatially correlated**, with **spatially varying magnitudes**, which is the focus of this module.

**Uncertainty** describes how well the true value is known, considering all types of error. In practice, the term
is often used loosely to refer to the **magnitude of random errors**, measured by the spread of their
distribution (commonly described by its standard deviation σ).

An **error structure**, short for error covariance structure, describes the random error magnitude at each pixel or
point, and their **correlation** to the error of every other location. For spatial problems, this structure can often be
simplified into a function depending only on separation distance. This description avoids storing a full spatial
covariance matrix, which would for instance contain **one trillion entries**
for a 1000 × 1000 raster.

The diagram below explains those concepts visually:

```{eval-rst}
.. plot:: diagrams/diagram_error_structure.py
    :width: 800px
```

(uncertainty-error-proxy)=
### Finding an error proxy

Uncertainty quantification often starts by finding a good **error proxy**, that is, a variable that can be used
to estimate the errors of interest.

Finding a suitable proxy is often application-specific, but common approaches include:

- **Coincident acquisitions:** Compare acquisitions at shared location/time with no expected change, including repeats
  with the same instrument. Their differences usually suffice to assess random errors, but shared systematic errors
  can cancel out, making those more difficult to identify.
- **Reference data:** Compare with independent ground measurements or data from a more precise instrument at shared
  locations. These differences help assess both systematic and random errors.
- **Stable surfaces:** Compare surveys from different dates over surfaces that have not changed.
  For elevation data, bare rock or grasslands can provide these reference areas.

If the error magnitudes and spatial correlations are already known, you can instead
{ref}`define an error structure from those parameters <uncertainty-model>`.

(uncertainty-error-predictors)=
### Predictors of error variability

A **predictor of error variability** is a measurement or surface attribute used to explain how errors vary.
It can help identify potential **systematic errors** or changes in **random error magnitude**. Examples include:

- **Measurement geometry:** Stereo convergence angle (optical), incidence angle (radar) or scan angle (lidar).
- **Measurement quality:** Image matching correlation (optical), interferometric coherence (radar) or
  signal-to-noise ratio (lidar).
- **Surface attributes:** Slope angle, surface roughness or land cover.

Use {ref}`grouped statistics <stats-grouped>` to examine how the mean and spread of the error proxy vary with these predictors.

(uncertainty-structure)=
## The error structure object

In GeoUtils, uncertainty features are centered around {class}`~geoutils.ErrorStructure`, a **lightweight** object that serves to
describe the error structure, without storing neither a full covariance matrix, nor raster/point-cloud predictors that might be associated to it.

An {class}`~geoutils.ErrorStructure` is composed of a sum of multiple **components** to describe errors from
different sources that can be assumed independent, such as separating local random noise from large-scale error patterns.
Each component has:

- An error **magnitude** model, describing its statistical spread. It can be constant or vary with any predictor, such as
  viewing angle or land cover type.
- An error **correlation** model, describing how errors relate across locations. Components can have
  different correlation ranges, or be spatially uncorrelated.

The figure below illustrates how an error structure **decomposes the total error into its components**, by
showing one realization of random error fields times their predicted magnitudes:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the plotting code on error components"
:  code_prompt_hide: "Hide the plotting code on error components"
:  image:
:    width: 800px
:    alt: "The snow depth error proxy combines a short range field scaled by viewing angle and land cover with an independent long range field scaled by a constant magnitude."

component_limit = np.nanmax(np.abs(quick_errors))
long_magnitude = quick_long_structure.predict_magnitude()

# Multiply each pattern by its magnitude, then add them
figure, axes = plt.subplots(
    1, 9, figsize=(13.5, 3.2), layout="constrained",
    gridspec_kw={"width_ratios": [1, 0.12, 1, 0.12, 1, 0.12, 1, 0.12, 1]},
)
panels = [
    (
        quick_noise, f"Short correlation range\n{quick_short_model.model.effective_range:g} m", "RdBu",
        -3, 3, "Standardized error",
    ),
    (
        quick_magnitude, "Variable magnitude\nfrom predictors", "YlOrRd",
        0, quick_magnitude.max(), "Error magnitude (m)",
    ),
    (
        quick_long_noise / long_magnitude, "Long correlation range\n220 m", "RdBu",
        -3, 3, "Standardized error",
    ),
    (
        np.full_like(quick_long_noise, long_magnitude), "Constant magnitude", "YlOrRd",
        0, quick_magnitude.max(), "Error magnitude (m)",
    ),
    (quick_errors, "Combined errors", "RdBu", -component_limit, component_limit, "Error (m)"),
]
for ax, (values, title, cmap, minimum, maximum, label) in zip(axes[::2], panels):
    image = ax.imshow(values, cmap=cmap, vmin=minimum, vmax=maximum, origin="upper")
    ax.set(title=title, xticks=[], yticks=[])
    figure.colorbar(image, ax=ax, orientation="horizontal", shrink=0.9, pad=0.15, label=label)

# Add the signs between maps
for ax, symbol in zip(axes[1::2], ["×", "+", "×", "="]):
    ax.set_axis_off()
    ax.text(0.5, 0.5, symbol, ha="center", va="center", fontsize=22, transform=ax.transAxes)
plt.show()
```

Structurally, an {class}`~geoutils.ErrorStructure` contains one or multiple {class}`~geoutils.ErrorComponent`,
each combining {class}`~geoutils.ErrorMagnitude` with an optional {class}`~geoutils.Variogram` correlation model. In practice,
**users should not need to manipulate those** subclasses during estimation, refitting or propagation, except for advanced use.
See {ref}`uncertainty-model` for manual construction.

These components can be summarized with {meth}`~geoutils.ErrorStructure.info`, or plotted with {meth}`~geoutils.ErrorStructure.plot` (all),
or {meth}`~geoutils.ErrorStructure.plot_magnitude` and {meth}`~geoutils.ErrorStructure.plot_correlation` for each aspect.

```{code-cell} ipython3
es.info()
```

An error structure is returned by {meth}`~geoutils.ErrorStructure.estimate` and passed to operations to
{ref}`uncertainty-propagation` or for {ref}`uncertainty-simulation`, which are both detailed directly below.

(uncertainty-estimation)=
## Estimate error structure

{meth}`geoutils.ErrorStructure.estimate`<br>
{meth}`ds.rst.estimate_error_structure() or Raster.estimate_error_structure() <RasterBase.estimate_error_structure>`<br>
{meth}`gdf.pc.estimate_error_structure() or PointCloud.estimate_error_structure() <PointCloudBase.estimate_error_structure>`<br>

For estimation, use {meth}`~geoutils.ErrorStructure.estimate` from an error structure object or {meth}`~RasterBase.estimate_error_structure`
from a raster or point cloud, interchangeably.

The sections below first explain the inputs and workflow to estimate an error structure, then how to choose
**error components** to refine estimation of their **magnitude** and **correlation**, and iterate until you have
satisfactorily modelled your error structure!

```{tip}
By default, error estimation uses the **standard deviation** as an estimator of spread for error magnitude, and
**Matheron's variogram** as an estimator of spatial correlation. Select the **NMAD** and **Dowd's variogram** instead for median-like estimators robust to outliers!
```

### Inputs and workflow

Estimating an error structure only strictly requires passing an **error proxy**, or a **reference data** to derive an
an error proxy by subtraction with the other data. Both can be rasters or point clouds.

The estimation workflow uses the following steps, all **scalable** on large datasets:

- First, {meth}`~RasterBase.cosample` selects valid data at common locations,
- Then, {meth}`~RasterBase.stats` estimates error magnitude, globally or grouped by predictors,
- Finally, {meth}`~RasterBase.variogram` estimates error correlation.

For example, reusing the synthetic snow depth data from the {ref}`uncertainty-quick-use`:

```{note}
When comparing to reference data, use `other_precision` to either neglect other errors (`"negligible"`, default), or
assume them independent with similar magnitude/correlation (`"same"`) which divides the error proxy by √2.
```

```{code-cell} ipython3
default_errors = ds.rst.estimate_error_structure(ds_ref)
default_errors.plot()
```

As we can see, **the fit is not convincing yet**.

By default, without defining **components** or **predictors**, GeoUtils uses one component with constant magnitude and a single-range spatial correlation model.
If **predictors** are passed, GeoUtils uses two components: a short range component with magnitude varying with predictors, plus a long range component with constant magnitude, both
with a single-range spatial correlation model.

The next sections explain how to define error **components** and potential **predictors** to reach a satisfactory estimation.

(uncertainty-constant-magnitude)=
(uncertainty-variable-magnitude)=
### Error magnitude

Error magnitude is **inferred by the spread of the error proxy** globally or within predictor groups,
directly relying on {ref}`stats-global` and {ref}`stats-grouped`.

By default, GeoUtils uses the **standard deviation** as spread estimator, with linear interpolation as model.
See {ref}`statistics estimators <stats-estimators>` for alternatives such as NMAD, and {ref}`uncertainty-custom-magnitude` for using any model.

#### Constant magnitude

A **constant magnitude** can be set to estimate the same average error spread everywhere, here we modify the estimator
into the **NMAD** (median-like standard deviation):

```{code-cell} ipython3
proxy = (ds - ds_ref).assign_attrs(units="m")

constant_errors = gu.ErrorStructure.estimate(
    proxy,
    components={"uncorrelated": {"magnitude": "constant", "correlation": None}},
    spread_estimator=gu.stats.nmad,
)
print(f"±{constant_errors.predict_magnitude():.2f} m (1σ)")
```

Here, `correlation=None` skips correlation estimation.
As shown below, the spread simply corresponds to that of all error proxy values:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the proxy and distribution"
:  code_prompt_hide: "Hide the code for plotting the proxy and distribution"
:  image:
:    width: 800px
:    alt: "The snow depth error proxy and its empirical distribution, with its NMAD and the mean ± NMAD interval."

# Compare the error map and its distribution
proxy_values = proxy.values.ravel()
proxy_values = proxy_values[np.isfinite(proxy_values)]
proxy_mean = np.mean(proxy_values)
proxy_nmad = constant_errors.predict_magnitude()
proxy_limit = np.max(np.abs(proxy_values))
figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
proxy.rst.plot(
    ax=axes[0], cmap="RdBu", vmin=-proxy_limit, vmax=proxy_limit,
    cbar_title="Error proxy (m)", title="Error proxy",
)
axes[0].set(xlabel="X (m)", ylabel="Y (m)")

axes[1].hist(proxy_values, bins=40, density=True, color="steelblue", alpha=0.7)
axes[1].axvspan(proxy_mean - proxy_nmad, proxy_mean + proxy_nmad, color="darkorange", alpha=0.2, label="Mean ± NMAD")
axes[1].axvline(proxy_mean, color="dimgray", linestyle="--", label="Mean")
axes[1].set(
    title=f"Error distribution\nNMAD = {proxy_nmad:.2f} m",
    xlabel="Error proxy (m)", ylabel="Probability density (1/m)",
)
axes[1].legend(frameon=False)
figure.tight_layout()
plt.show()
```

#### One predictor

To infer **variable magnitudes** (also called **heteroscedasticity**), we bin the error proxy by predictor values to get enough samples to estimate
its spread (we assume similar predictor values have similar error spread elsewhere).

For example, below we use **viewing angle** to group spread estimates, selecting `"heteroscedastic"` for a variable magnitude,
and plotting it with {meth}`~geoutils.ErrorStructure.plot_magnitude`:

```{code-cell} ipython3
magnitude_components = {"uncorrelated": {"magnitude": "heteroscedastic", "correlation": None}}
angle_errors = gu.ErrorStructure.estimate(
    proxy,
    predictors={"view_angle": view_angle},
    components=magnitude_components,
    bins=6,
)

```

We can then use {meth}`~geoutils.ErrorStructure.predict_magnitude` to predict the error for any view angle value, typically
useful to create a per-pixel/point error map with the shape of the input:

```{code-cell} ipython3
magnitude_map = angle_errors.predict_magnitude({"view_angle": view_angle}, like=ds)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the predictor, bins and predictions"
:  code_prompt_hide: "Hide the code for plotting the predictor, bins and predictions"
:  image:
:    width: 800px
:    alt: "Viewing angle, error magnitude and sample counts in six viewing angle bins, and predicted error magnitudes across the snow depth grid."

# Show viewing angle, binned magnitudes and predictions
figure, axes = plt.subplots(1, 3, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.35, 1]})
view_angle.rst.plot(ax=axes[0], cmap="viridis", cbar_title="Viewing angle (°)", title="Viewing angle")
magnitude_map.rst.plot(
    ax=axes[2], cmap="YlOrRd", cbar_title="Error magnitude (m)", title="Predicted error magnitude",
)
for ax in [axes[0], axes[2]]:
    ax.set(xlabel="X (m)", ylabel="Y (m)")

# Leave room for the bin counts
figure.subplots_adjust(left=0.06, right=0.93, bottom=0.2, top=0.85, wspace=0.7)
angle_errors.plot_magnitude(ax=axes[1], title="Binned error magnitude")
plt.show()
```

For a categorical predictor like **land cover** instead of a continuous one, use:

```{code-cell} ipython3
cover_errors = gu.ErrorStructure.estimate(
    proxy,
    predictors={"land_cover": land_cover},
    components=magnitude_components,
    categories={"land_cover": cover_codes},
)
cover_magnitude_map = cover_errors.predict_magnitude({"land_cover": land_cover}, like=ds)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the predictor, bins and predictions"
:  code_prompt_hide: "Hide the code for plotting the predictor, bins and predictions"
:  image:
:    width: 800px
:    alt: "Land cover, error magnitude and sample counts for each category, and predicted error magnitudes across the snow depth grid."

# Show land cover, binned magnitudes and predictions
figure, axes = plt.subplots(1, 3, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.35, 1]})
land_cover.rst.plot(
    ax=axes[0], cmap=ListedColormap(cover_colors),
    vmin=-0.5, vmax=2.5, add_cbar=False, title="Land cover",
)
axes[0].legend(handles=legend_handles, loc="upper right", framealpha=0.9)
cover_magnitude_map.rst.plot(
    ax=axes[2], cmap="YlOrRd", cbar_title="Error magnitude (m)", title="Predicted error magnitude",
)
for ax in [axes[0], axes[2]]:
    ax.set(xlabel="X (m)", ylabel="Y (m)")

# Leave room for the bin counts
figure.subplots_adjust(left=0.06, right=0.93, bottom=0.2, top=0.85, wspace=0.7)
cover_errors.plot_magnitude(ax=axes[1], title="Binned error magnitude")
plt.show()
```

#### Two predictors

Multiple predictors can be combined at once, grouping bins/categories together as in {meth}`~RasterBase.stats`.

Below, we use the **viewing angle and land cover** simultaneously:

```{code-cell} ipython3
variable_errors = gu.ErrorStructure.estimate(
    proxy,
    predictors={"view_angle": view_angle, "land_cover": land_cover},
    components=magnitude_components,
    bins={"view_angle": 6},
    categories={"land_cover": cover_codes},
)
axes = variable_errors.plot_magnitude()
```

Use {meth}`~geoutils.ErrorStructure.refit_magnitude` to fit **another model** (which default to linear interpolation),
using the same binned estimates. Changing bins, categories or estimator requires running {meth}`~geoutils.ErrorStructure.estimate` again.

Parametric models can also be used, as demonstrated in {ref}`uncertainty-custom-magnitude`.

(uncertainty-correlation)=
### Error correlation

Error correlation is **inferred by the spatial correlation of the error proxy**, relying on {ref}`stats-spatial`.

By default, GeoUtils uses the **Matheron** variogram estimator and fits a single-range **Gaussian** model per component.
See {ref}`variogram estimators <stats-spatial-estimators>` for alternatives, and {ref}`variogram fitting <stats-spatial-fitting>` for using any model form.

#### Estimator and model

We set a model form and variogram estimator, and plot with {meth}`~geoutils.ErrorStructure.plot_correlation`.

```{code-cell} ipython3
:mystnb:
:  image:
:    width: 600px

# Use a spherical model
single_errors = gu.ErrorStructure.estimate(
    proxy,
    components={"correlated": {"magnitude": "constant", "correlation": "spherical"}},
    variogram_estimator="dowd",
    n_pairs=100_000,
    n_runs=3,
    pair_sampling_kwargs={"strategy": "independent"},
    random_state=42,
)
ax = single_errors.plot_correlation(kind="correlation", title="One correlation model")
```

Here, a single model **does not fit empirical correlation reliably**, so we need models that can
account for multiple correlation ranges, as examplified directly below.

```{note}
Semivariance increases as correlation decreases, following {math}`\rho(h) = 1 - \gamma(h)/C(0)`, where {math}`C(0)`
is the variance (sill). Use `kind="semivariance"` to plot semivariance instead.
```

#### Multiple-range models

It is often useful to fit a **sum of models** to represent both a short and a long correlation range.

For instance:
```{code-cell} ipython3
:mystnb:
:  image:
:    width: 600px

correlated_errors = gu.ErrorStructure.estimate(
    proxy,
    components={
        "short_range": {"magnitude": "constant", "correlation": "gaussian"},
        "long_range": {"magnitude": "constant", "correlation": "gaussian"},
    },
    variogram_estimator="matheron",
    n_pairs=100_000,
    n_runs=3,
    pair_sampling_kwargs={"strategy": "independent"},
    random_state=42,
)
ax = correlated_errors.plot_correlation(kind="correlation", title="Two correlation models")
```

To **refit spatial correlation**, use {meth}`~geoutils.ErrorStructure.refit_variogram` with different models, which reuses the stored empirical variogram.
Changing empirical variogram estimation requires running {meth}`~geoutils.ErrorStructure.estimate` again.

```{code-cell} ipython3
refitted_errors = correlated_errors.refit_variogram(["gaussian", "exponential"])
```

Custom variogram functions are described in {ref}`uncertainty-custom-variogram`.

(uncertainty-propagation)=
## Propagate uncertainty

{func}`geoutils.uncertainty.propagate() <geoutils.uncertainty.propagate>`<br>

Once an error structure is estimated, **propagation** computes uncertainty through a given operation.

This can be done either **analytically**, or **numerically**. Both return a {class}`~geoutils.PropagationSummary` object summarizing the propagation.
Alternatively, you can directly generate **random fields** from the above error structure, respecting multiple correlations and variable errors,
to perform your own simulations.

(uncertainty-propagation-summary)=
### Propagation summary

A {class}`~geoutils.PropagationSummary` is an object that stores metrics about the output of the uncertainty propagation.

```{important}
Currently, propagation outputs are not designed to provide an {class}`~geoutils.ErrorStructure` for the output.
If you need to keep propagating errors of your output to another step, **bundle all steps at once** during your propagation.
```

For example, we propagate uncertainty through the **mean snow depth of 8 × 8 cells**.
The summary stores the original result in {attr}`~geoutils.PropagationSummary.estimate`, its **1σ uncertainty**
in {attr}`~geoutils.PropagationSummary.std`, and its mean in {attr}`~geoutils.PropagationSummary.mean`:

```{note}
A propagated {attr}`~geoutils.PropagationSummary.estimate` is the value from the original input, whereas {attr}`~geoutils.PropagationSummary.mean` averages the simulated results. They can differ for nonlinear
calculations.
```

```{code-cell} ipython3
summary = gu.uncertainty.propagate(
    ds.rst.icrop((0, 0, 8, 8)).rst.reproject,
    error_structure=correlated_errors,
    method="analytical",
    operation_kwargs={"grid_size": (1, 1), "resampling": "average"},
)
summary.estimate.item(), summary.std.item(), summary.mean.item()
```

Use {meth}`~geoutils.PropagationSummary.info` for a summary of the propagation:

```{code-cell} ipython3
summary.info()
```

And {meth}`~geoutils.PropagationSummary.interval` for estimating **95% uncertainty bounds**:

```{code-cell} ipython3
summary.interval(0.95)
```

These attributes have the **same shape as the operation's output**. For a raster or point cloud, they also share the output coordinates, with an uncertainty for each cell or point.

For example, if resample to a **30 m grid**, propagation returns three rasters:

```{code-cell} ipython3
raster_summary = gu.uncertainty.propagate(
    ds.rst.reproject,
    error_structure=correlated_errors,
    method="analytical",
    operation_kwargs={"res": 30, "resampling": "average"},
)
raster_summary.estimate.shape, raster_summary.mean.shape, raster_summary.std.shape
```

(uncertainty-analytical)=
### Analytical propagation

Analytical propagation is only supported for **linear calculations**, using the calculation weights with the error covariance. See {ref}`core-operators` for details on how to
define a linear operator in GeoUtils.

For instance, we compute the mean of **8 × 8 cells** and propagate its uncertainty:

```{code-cell} ipython3
mean_value, mean_summary = proxy.rst.icrop((0, 0, 8, 8)).rst.stats(
    "mean", error_structure=correlated_errors,
    uncertainty_kwargs={"method": "analytical"},
)
print(f"Mean: {mean_value:.2f} ± {mean_summary.std.item():.2f} m (1σ)")
```

Then, we can repeat this mean over larger contiguous squares to see how uncertainty decreases with area:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting uncertainty against area"
:  code_prompt_hide: "Hide the code for plotting uncertainty against area"
:  image:
:    width: 800px
:    alt: "Nested square averaging areas and their mean uncertainties, which decrease more slowly for correlated errors than for uncorrelated errors of the same magnitude."

from matplotlib.patches import Rectangle

# Average larger squares
window_sizes = np.array([1, 3, 5, 7, 9, 13, 17, 25, 33])
center_row = proxy.rst.height // 2
center_column = proxy.rst.width // 2
window_uncertainties = []
for size in window_sizes:
    half_size = size // 2
    bounds = (
        center_column - half_size, center_row - half_size,
        center_column + half_size + 1, center_row + half_size + 1,
    )
    window = proxy.rst.icrop(bounds)
    _, window_summary = window.rst.stats(
        "mean", error_structure=correlated_errors,
        uncertainty_kwargs={"method": "analytical"},
    )
    window_uncertainties.append(window_summary.std.item())

# Mark a few squares on the map
pixel_size = abs(proxy.rst.transform.a)
center_x, center_y = proxy.rst.ij2xy(center_row, center_column)
context = proxy.rst.icrop((center_column - 19, center_row - 19, center_column + 20, center_row + 20))
figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
context.rst.plot(ax=axes[0], cmap="RdBu", cbar_title="Error proxy (m)")
for size in [5, 13, 33]:
    side_length = size * pixel_size
    lower_left = (center_x - side_length / 2, center_y - side_length / 2)
    axes[0].add_patch(Rectangle(lower_left, side_length, side_length, fill=False, edgecolor="black"))
    axes[0].text(
        center_x, center_y + side_length / 2, f"{size} × {size} cells",
        ha="center", va="bottom", fontsize=8, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
    )
axes[0].set(title="Contiguous averaging areas", xlabel="X (m)", ylabel="Y (m)")

# Compare correlated and independent errors
areas = (window_sizes * pixel_size) ** 2
source_magnitude = correlated_errors.predict_magnitude()
axes[1].loglog(areas, window_uncertainties, "o-", color="darkorange", label="Spatially correlated")
axes[1].loglog(areas, source_magnitude / window_sizes, "--", color="steelblue", label="Uncorrelated: σ / √n")
axes[1].set(
    title="Uncertainty of the area mean", xlabel="Averaging area (m²)", ylabel="Mean uncertainty (m)",
)
axes[1].grid(alpha=0.2)
axes[1].legend(frameon=False)
figure.tight_layout()
plt.show()
```

```{tip}
Spatial correlation makes uncertainty decrease more slowly when averaging cells.
The familiar standard error {math}`\sigma / \sqrt{n}` rule only works for independent errors with equal magnitudes.
```

To further examplify this, we propagate uncertainty through {meth}`~RasterBase.reproject` **from a 10 m grid to a 30 m grid**.
The supplied error covariance both **weights the average resampling**, and calculates uncertainty for each output cell:

```{code-cell} ipython3
# Match predictor values to source cell IDs in row order
cell_predictors = {
    name: dict(enumerate(np.asarray(values).ravel()))
    for name, values in predictors.items()
}

summary = gu.uncertainty.propagate(
    ds.rst.reproject,
    error_structure=es,
    predictors=cell_predictors,
    method="analytical",
    operation_kwargs={"res": 30, "resampling": "average"},
)
source_uncertainty = es.predict_magnitude(predictors, like=ds)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for comparing the original and reprojected maps"
:  code_prompt_hide: "Hide the code for comparing the original and reprojected maps"
:  image:
:    width: 800px
:    alt: "Snow depth above its random error magnitude, before and after weighted averaging from a 10 m to a 30 m grid. Plus-minus signs link each estimate to its error magnitude, with a resampling arrow between them."

# Compare snow depth and error magnitude on shared scales
snow_limits = (ds.min().item(), ds.max().item())
uncertainty_limit = source_uncertainty.max().item()
figure, axes = plt.subplots(
    3, 3, figsize=(9, 7.5), layout="constrained",
    gridspec_kw={"width_ratios": [1, 0.24, 1], "height_ratios": [1, 0.12, 1]},
)
panels = [
    (axes[0, 0], ds, "Snow depth before\n10 m grid", "viridis", *snow_limits),
    (axes[0, 2], summary.estimate, "Snow depth after weighted averaging\n30 m grid", "viridis", *snow_limits),
    (axes[2, 0], source_uncertainty, "Error magnitude before", "YlOrRd", 0, uncertainty_limit),
    (axes[2, 2], summary.std, "Propagated error magnitude", "YlOrRd", 0, uncertainty_limit),
]
for ax, values, title, cmap, minimum, maximum in panels:
    values.rst.plot(ax=ax, cmap=cmap, vmin=minimum, vmax=maximum, add_cbar=False)
    ax.set(title=title, xlabel="X (m)", ylabel="Y (m)")

# Add colorbars for each row
for ax in [*axes[1], axes[0, 1], axes[2, 1]]:
    ax.set_axis_off()
figure.colorbar(axes[0, 0].images[0], ax=axes[0, [0, 2]], shrink=0.85, label="Snow depth (m)")
figure.colorbar(axes[2, 0].images[0], ax=axes[2, [0, 2]], shrink=0.85, label="Error magnitude (m, 1σ)")

# Align the signs and arrow after placing the colorbars
figure.canvas.draw()
map_centers = [ax.get_position().x0 + ax.get_position().width / 2 for ax in axes[0, [0, 2]]]
symbol_y = axes[1, 1].get_position().y0 + axes[1, 1].get_position().height / 2
for center in map_centers:
    figure.text(center, symbol_y, "±", ha="center", va="center", fontsize=24)
arrow_left = axes[0, 0].get_position().x1 - 0.015
arrow_right = axes[0, 2].get_position().x0 + 0.015
arrow = axes[1, 1].annotate(
    "", (arrow_right, symbol_y), (arrow_left, symbol_y),
    xycoords="figure fraction", arrowprops={"arrowstyle": "->", "lw": 1.5},
)
arrow.set_in_layout(False)
figure.text(
    (arrow_left + arrow_right) / 2, symbol_y + 0.008, "Resampling",
    ha="center", va="bottom", fontsize=10,
)
plt.show()
```

(uncertainty-numerical-intervals)=
### Numerical propagation

Numerical propagation is supported for **any operation**, but computationally intensive as it repeats realizations of the calculation with random errors added to the inputs.
These realizations are generated through {ref}`uncertainty-simulation`.

```{code-cell} ipython3
numerical_window = proxy.rst.icrop((4, 4, 12, 12))
median_value, median_summary = numerical_window.rst.stats(
    "median", error_structure=constant_errors,
    uncertainty_kwargs={"method": "numerical", "n_simulations": 500, "return_samples": True, "random_state": 42},
)
median_uncertainty = median_summary.std.item()
print(f"Median: {median_value:.2f} ± {median_uncertainty:.2f} m (1σ)")
```

The maps below show the integration window and two error realizations. Adding each realization to the same input
window and calculating its median produces one value in the distribution on the right:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the numerical simulations"
:  code_prompt_hide: "Hide the code for plotting the numerical simulations"
:  image:
:    width: 800px
:    alt: "An 8 by 8 integration window, two simulated error fields and their medians, and 500 simulated medians with the original median ± one standard deviation. Black ticks labelled 1 and 2 locate the two draws."

# Repeat the first two simulations
realizations = numerical_window.rst.random_field(
    constant_errors, n_fields=2, random_state=42,
)
realization_medians = [(numerical_window + realization).rst.stats("median") for realization in realizations]
figure, axes = plt.subplots(
    1, 4, figsize=(12, 3.6), layout="constrained", gridspec_kw={"width_ratios": [1, 1, 1, 1.5]},
)

# Outline the area used for the median
numerical_context = proxy.rst.icrop((0, 0, 16, 16))
numerical_context.rst.plot(ax=axes[0], cmap="viridis", add_cbar=False)
window_x = numerical_window.rst.bbox.left
window_y = numerical_window.rst.bbox.bottom
window_side = numerical_window.rst.width * pixel_size
axes[0].add_patch(
    Rectangle((window_x, window_y), window_side, window_side, fill=False, edgecolor="white", linewidth=2),
)
axes[0].set(title=f"Input values\n8 × 8 cells ({window_side:g} × {window_side:g} m)", xlabel="X (m)", ylabel="Y (m)")

# Plot source errors with a shared scale
error_limit = 3 * constant_errors.predict_magnitude()
for number, (ax, realization, value) in enumerate(zip(axes[1:3], realizations, realization_medians), 1):
    image = ax.imshow(realization, cmap="RdBu", vmin=-error_limit, vmax=error_limit)
    ax.set(title=f"Error draw {number}\nResulting median: {value:.2f} m", xticks=[], yticks=[])
figure.colorbar(image, ax=axes[1:3], orientation="horizontal", shrink=0.8, label="Source error (m)")

# Show the spread of the simulated medians
axes[3].axvspan(
    median_value - median_uncertainty, median_value + median_uncertainty,
    color="steelblue", alpha=0.12, label="Median ±1σ",
)
axes[3].hist(median_summary.samples.iloc[:, 0], bins=30, density=True, color="steelblue", alpha=0.6)
axes[3].axvline(median_value, color="dimgray", label="Original median")
for number, value in enumerate(realization_medians, 1):
    axes[3].plot(value, 0, "|", color="black", markersize=14, markeredgewidth=2, clip_on=False, zorder=5)
    axes[3].annotate(
        str(number), (value, 0), xytext=(0, 12), textcoords="offset points", ha="center",
        color="black", fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
    )
axes[3].set(
    title=f"500 repeated medians\nUncertainty (1σ): {median_uncertainty:.2f} m",
    xlabel="Window median (m)", ylabel="Probability density (1/m)",
)
axes[3].legend(frameon=False, fontsize=8)
plt.show()
```

(uncertainty-simulation)=
### Random field generation

**Random error fields** are realizations of an error structure, generated with {meth}`~RasterBase.random_field`:

```{code-cell} ipython3
field = ds.rst.random_field(
    es, predictors={"view_angle": view_angle, "land_cover": land_cover}, backend="gstools", random_state=42,
)
```

```{note}
Random field generation **relies on [GSTools](https://geostat-framework.readthedocs.io/projects/gstools/en/stable/) or [GPyTorch](https://docs.gpytorch.ai/en/stable/)**. The former supports more correlation forms, with the latter is usually faster.

GeoUtils adds logic to **directly support complex error structure** from multiple-range correlation models and variable magnitude from predictors, applicable on large datasets.
```

The examples below emphasize variable magnitudes and increasingly long correlation ranges:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for comparing random fields"
:  code_prompt_hide: "Hide the code for comparing random fields"
:  image:
:    width: 1000px
:    alt: "Five random fields with constant or variable magnitude: uncorrelated errors, short range correlation, and several short and long correlation ranges."

import pandas as pd

# Make errors grow with viewing angle
field_statistics = pd.DataFrame(
    {"std": [0.03, 0.6], "count": [100, 100]},
    index=pd.Index([0.0, 60.0], name="view_angle"),
)
variable_magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(field_statistics)
variable_model = gu.ErrorStructure([gu.ErrorComponent("uncorrelated", variable_magnitude)])
field_magnitudes = variable_model.predict_magnitude({"view_angle": view_angle}, like=ds)
field_magnitude = np.sqrt((field_magnitudes ** 2).mean()).item()

# Use the same average error variance in all five examples
field_short_model = gu.Variogram.from_model("gaussian", effective_range=100, partial_sill=1)
uncorrelated_model = gu.ErrorStructure([gu.ErrorComponent("uncorrelated", field_magnitude)])
correlated_model = gu.ErrorStructure([gu.ErrorComponent("correlated", field_magnitude, field_short_model)])
variable_correlated_model = gu.ErrorStructure([gu.ErrorComponent("correlated", variable_magnitude, field_short_model)])
range_models = [
    gu.Variogram.from_model("gaussian", effective_range=distance, partial_sill=variance)
    for distance, variance in [(25, 0.1), (100, 0.3), (400, 0.6)]
]
mixed_correlation = gu.Variogram.combine(*range_models)
mixed_model = gu.ErrorStructure([gu.ErrorComponent("correlated", variable_magnitude, mixed_correlation)])
uncorrelated_field = ds.rst.random_field(uncorrelated_model, random_state=42)
variable_uncorrelated_field = ds.rst.random_field(variable_model, predictors={"view_angle": view_angle}, random_state=42)
correlated_field = ds.rst.random_field(correlated_model, backend="gstools", random_state=42)
variable_correlated_field = ds.rst.random_field(
    variable_correlated_model, predictors={"view_angle": view_angle}, backend="gstools", random_state=42,
)
mixed_field = ds.rst.random_field(
    mixed_model, predictors={"view_angle": view_angle}, backend="gstools", random_state=42,
)

# Use one color scale for all five maps
comparison_fields = [uncorrelated_field, variable_uncorrelated_field, correlated_field, variable_correlated_field, mixed_field]
field_limit = 3 * field_magnitude
figure, axes = plt.subplots(1, 5, figsize=(12, 4), sharex=True, sharey=True, layout="compressed")
figure.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.02)
for ax, values, title in zip(
    axes, comparison_fields,
    [
        "Uncorrelated\nConstant magnitude", "Uncorrelated\nVariable magnitude",
        "Correlated short range\nConstant magnitude", "Correlated short range\nVariable magnitude",
        "Several short and long ranges\nVariable magnitude",
    ],
):
    values.rst.plot(ax=ax, cmap="RdBu", vmin=-field_limit, vmax=field_limit, add_cbar=False)
    ax.set(title=title, xlabel="X (m)")
axes[0].set_ylabel("Y (m)")
figure.colorbar(
    axes[0].images[0], ax=axes, orientation="horizontal", shrink=0.7, aspect=40,
    fraction=0.06, pad=0.08, extend="both", label="Error (m)",
)
plt.show()
```

The field contains **errors only**. Add it to your data to create perturbed inputs for simulations:

```{code-cell} ipython3
ds_simulated = ds + field
```

(uncertainty-advanced)=
## Advanced usage

The following sections present more advanced usage of the uncertainty features, such as **defining your
error structure** manually, or **fitting custom models** for magnitude/correlation.

(uncertainty-custom-magnitude)=
### Custom magnitude fit

One can pass **a custom function** to fit the error magnitude bins, either as
`magnitude_model` and `magnitude_fit_kwargs` arguments to {meth}`~geoutils.ErrorStructure.estimate`, or to
{meth}`~geoutils.ErrorStructure.refit_magnitude` more directly:

```{code-cell} ipython3
def angle_spread(angle, offset, quadratic):
    """Calculate error magnitude as a quadratic function of viewing angle."""
    return offset + quadratic * angle**2


angle_model = angle_errors.refit_magnitude(
    angle_spread,
    fit_kwargs={"p0": [0.1, 0.001], "bounds": (0, float("inf"))},
)
axes = angle_model.plot_magnitude()
```

(uncertainty-custom-variogram)=
### Custom variogram fit

One can pass **a custom function** to fit the variogram lag bins, either as a component's `correlation` with
`fit_kwargs` in {meth}`~geoutils.ErrorStructure.estimate`, or to
{meth}`~geoutils.ErrorStructure.refit_variogram` more directly:

```{code-cell} ipython3
def stretched_variogram(distance, effective_range, partial_sill, exponent):
    """Calculate semivariance with a fitted exponent controlling the correlation decrease."""
    return partial_sill * (1 - np.exp(-3 * (distance / effective_range)**exponent))


custom_correlation = single_errors.refit_variogram(
    stretched_variogram,
    fit_kwargs={
        "p0": [80, 1, 1],
        "bounds": [(1, 400), (0, float("inf")), (0.2, 2)],
    },
)
ax = custom_correlation.plot_correlation(kind="correlation")
```

The **exponent bounds** keep the covariance valid.

(uncertainty-model)=
### Definition from known parameters

An error structure can also be **defined from known parameters** using {class}`~geoutils.ErrorComponent`.
Each component combines an {class}`~geoutils.ErrorMagnitude` with an optional {class}`~geoutils.Variogram`
correlation model.
Here, spatially uncorrelated and correlated components have a **combined variance of 5 m²** (their variances add):

```{code-cell} ipython3
:mystnb:
:  image:
:    width: 800px

from geoutils.stats.variography import VariogramModel

uncorrelated = gu.ErrorComponent("uncorrelated", magnitude=1.0)
correlation = VariogramModel("gaussian", effective_range=100, partial_sill=1)
correlated = gu.ErrorComponent("correlated", magnitude=2.0, correlation=correlation)
known_errors = gu.ErrorStructure([uncorrelated, correlated])
known_errors.plot(show_components=True)
```
