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
(uncertainty)=
# Uncertainty

GeoUtils implements tools to **estimate uncertainty** and **propagate its effect** on spatial calculations.

We focus on measurement errors whose **magnitudes vary spatially** or that are **spatially correlated**, for either rasters 
or point clouds.

GeoUtils does not handle correlated errors between different variables, such as an instrument's bands. For these, see [ObsArray](https://obsarray.readthedocs.io/en/latest/).

Two steps describe the workflow:

- {ref}`Estimate an error structure <uncertainty-estimation>` from measurement differences, using
  {ref}`grouped statistics <stats-grouped>` for error magnitudes and {ref}`variograms <stats-spatial>` for spatial
  correlations, as described on the {ref}`Statistics page <stats>`.
- Reuse it to **weight measurements** or {ref}`propagate uncertainty <uncertainty-propagation>` through different
  operations, **numerically** with {ref}`random field simulations <uncertainty-simulation>` or **analytically** where
  supported.

```{caution}
The uncertainty API is new and **experimental**. It may change in future releases.
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
:  code_prompt_show: "Show the imports"
:  code_prompt_hide: "Hide the imports"

import numpy as np
from affine import Affine

import geoutils as gu
from geoutils.operators import GridNeighbours, LocalData
from geoutils.operators.reducer import Mean, Median
from geoutils.stats.variography import VariogramModel
```

(uncertainty-estimation)=
## Estimating errors from measurements

An **error proxy** contains values that represent measurement errors, such as differences between repeated
measurements of an unchanged surface. Real changes and systematic offsets must be considered separately before
using those differences to estimate random errors.

### Constant magnitude

{meth}`ds.rst.estimate_error_structure() or Raster.estimate_error_structure() <RasterBase.estimate_error_structure>`<br>
{meth}`gdf.pc.estimate_error_structure() or PointCloud.estimate_error_structure() <PointCloudBase.estimate_error_structure>`

Compare two measurements of the same quantity to **estimate their error magnitude**. Here, we measure a constant
100 m surface with independent errors of **2 m standard deviation**, and compare it with a precise reference.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for creating synthetic measurements"
:  code_prompt_hide: "Hide the code for creating synthetic measurements"

# Create repeated measurements of a constant surface with independent 2 m errors
random = np.random.default_rng(42)
grid = Affine(10, 0, 0, 0, -10, 640)
reference_values = np.full((64, 64), 100.0)
reference = gu.raster.from_array(reference_values, grid, 32631)
measured = gu.raster.from_array(
    reference_values + random.normal(0, 2, reference_values.shape),
    grid,
    32631,
)
```

```{code-cell} ipython3
# Treat the reference as precise and estimate only independent measurement errors
estimated_errors = measured.rst.estimate_error_structure(
    reference,
    other_precision="negligible",
    components={"measurement": {"magnitude": "constant", "correlation": None}},
    spread_estimator=np.std,
    random_state=42,
)
round(estimated_errors.predict_magnitude(), 2)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"
:  image:
:    alt: "Measurement differences follow a normal distribution with known 2 m and fitted 1.99 m error magnitudes."

# Compare the error proxy's histogram with the known and estimated normal distributions
from scipy.stats import norm

proxy_values = np.asarray(measured - reference).ravel()
error_values = np.linspace(-7, 7, 300)
figure, ax = plt.subplots(figsize=(6, 3), layout="constrained")
ax.hist(
    proxy_values,
    bins=35,
    density=True,
    color="#3975a7",
    alpha=0.45,
    label="Measurement differences",
)
ax.plot(
    error_values,
    norm.pdf(error_values, scale=2),
    color="0.25",
    label="Known magnitude: 2.00 m",
)
estimated_magnitude = estimated_errors.predict_magnitude()
ax.plot(
    error_values,
    norm.pdf(error_values, scale=estimated_magnitude),
    "--",
    color="#e38b35",
    label=f"Estimated magnitude: {estimated_magnitude:.2f} m",
)
ax.set(
    xlabel="Error (m)",
    ylabel="Probability density (1/m)",
    title="Estimate the spread of measurement errors",
)
ax.legend(frameon=False)
plt.show()
```

The fitted magnitude is close to the known **2 m**. The default spread estimator,
{func}`~geoutils.stats.nmad`, is less sensitive to outliers than `np.std`. The normal curves illustrate the synthetic
data; a magnitude estimate alone does not establish an error distribution.

```{important}
Choose `other_precision="negligible"` when the reference's uncertainty can be ignored. The default,
`other_precision="same"`, assumes both measurements have independent errors with the same magnitude and correlation,
and divides their difference by `sqrt(2)`. These assumptions must be appropriate for your measurements.
```

### Magnitudes that vary with predictors

Errors can vary with terrain, instrument geometry or other predictors. A **heteroscedastic** component fits the
spread within predictor groups and interpolates that magnitude at new predictor values.

Here, the known magnitude increases from **1 to 3 m** as the sensor viewing angle increases from 0 to 60°.
{meth}`~geoutils.ErrorStructure.estimate` accepts a proxy directly, without subtracting a second measurement.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for creating a variable error proxy"
:  code_prompt_hide: "Hide the code for creating a variable error proxy"

# Create an error proxy whose spread increases with a viewing angle
view_angle_values = np.broadcast_to(np.linspace(0, 60, 64), (64, 64)).copy()
view_angle = gu.raster.from_array(view_angle_values, grid, 32631)
proxy = gu.raster.from_array(
    (1 + view_angle_values / 30) * random.normal(size=view_angle_values.shape),
    grid,
    32631,
)
```

```{code-cell} ipython3
# Fit grouped magnitudes without a spatial correlation component
variable_errors = gu.ErrorStructure.estimate(
    proxy,
    predictors={"view_angle": view_angle},
    components={"measurement": {"magnitude": "heteroscedastic", "correlation": None}},
    bins=8,
    min_count=100,
    spread_estimator=np.std,
    random_state=42,
)
magnitude_map = variable_errors.predict_magnitude({"view_angle": view_angle}, like=proxy)
np.round(variable_errors.predict_magnitude({"view_angle": np.array([15.0, 45.0])}), 2)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"
:  image:
:    alt: "Error spread increases with the viewing angle, and the fitted magnitude follows the known trend."

# Show the error spread, the fitted trend and its spatial prediction
figure, axes = plt.subplots(1, 3, figsize=(10, 3))
axes[0].scatter(
    view_angle_values.ravel(),
    np.asarray(proxy).ravel(),
    s=2,
    alpha=0.12,
    color="#3975a7",
    rasterized=True,
)
axes[0].set(title="Error proxy", xlabel="Viewing angle (°)", ylabel="Error (m)")

predictor_values = np.linspace(5, 55, 100)
fitted_magnitudes = variable_errors.predict_magnitude({"view_angle": predictor_values})
axes[1].plot(
    predictor_values, 1 + predictor_values / 30, color="0.25", label="Known magnitude"
)
axes[1].plot(
    predictor_values, fitted_magnitudes, color="#e38b35", label="Fitted magnitude"
)
axes[1].set(
    title="Known and fitted magnitudes",
    xlabel="Viewing angle (°)",
    ylabel="Error magnitude (m)",
    ylim=(0.8, 3.2),
)
axes[1].legend(frameon=False)
magnitude_map.rst.plot(
    ax=axes[2], cmap="YlOrRd", vmin=1, vmax=3, cbar_title="Error magnitude (m)"
)
axes[2].set(title="Predicted error magnitude", xlabel="X (m)", ylabel="Y (m)")
plt.tight_layout()
plt.show()
```

`predict_magnitude()` evaluates the model at new predictor values. With `like=proxy`, it returns a raster on the
proxy's grid. Use `plot_magnitude()` to inspect the fitted groups and their sample counts.

(uncertainty-correlation)=
### Spatial correlation

A magnitude alone does not describe how errors relate across locations. Specify a correlation such as
`"spherical"` to **fit it from a variogram of the error proxy**. Below, the synthetic proxy has a **2 m magnitude**
and a **100 m spherical correlation range**.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for creating a correlated error proxy"
:  code_prompt_hide: "Hide the code for creating a correlated error proxy"

# Create a proxy with known spatial correlation on the measurement grid
known_correlation = VariogramModel("spherical", effective_range=100, partial_sill=1)
known_errors = gu.ErrorStructure(
    [gu.ErrorComponent("spatial", magnitude=2.0, correlation=known_correlation)]
)
spatial_proxy = measured.rst.random_field(
    known_errors, backend="gstools", random_state=42
)
```

```{code-cell} ipython3
# Fit the magnitude and correlation using a sample of nearby and distant pairs
correlated_errors = gu.ErrorStructure.estimate(
    spatial_proxy,
    components={"spatial": {"magnitude": "constant", "correlation": "spherical"}},
    spread_estimator=np.std,
    variogram_estimator="matheron",
    n_pairs=20_000,
    n_lags=12,
    max_lag=250,
    random_state=42,
)
round(correlated_errors.components["spatial"].correlation.effective_range, 1)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"
:  image:
:    alt: "A correlated error map and its standardized variogram, comparing sampled pairs with fitted and known models."

# Compare the simulated proxy with its empirical variogram and fitted correlation range
figure, axes = plt.subplots(1, 2, figsize=(7, 3.2))
spatial_proxy.rst.plot(ax=axes[0], cmap="RdBu", vmin=-6, vmax=6, cbar_title="Error (m)")
axes[0].set_title("Correlated error proxy")
axes[0].set(xlabel="X (m)", ylabel="Y (m)")
correlated_errors.plot_correlation(ax=axes[1], show_counts=False, xscale="linear")
lag_distances = np.linspace(0, 250, 300)
axes[1].plot(
    lag_distances,
    known_correlation.variogram(lag_distances),
    "--",
    color="#e38b35",
    label="Known normalized model",
)
axes[1].set(
    title="Standardized error variogram",
    xlabel="Separation (m)",
    ylabel="Semivariance (unitless)",
)
axes[1].legend(frameon=False)
plt.tight_layout()
plt.show()
```

Nearby errors are similar, so their semivariance is small; it rises toward a plateau as separation increases.
`plot_correlation()` compares the sampled pairs with the fitted model. GeoUtils first divides the proxy by its
magnitude, so this variogram is **dimensionless**. A single simulated field only approximately recovers the known model.

(uncertainty-model)=
### Defining an error structure from known parameters

The estimates above return an {class}`~geoutils.ErrorStructure`. It combines named
{class}`~geoutils.ErrorComponent` objects, each with an **error magnitude** and an optional **spatial correlation
model**. The magnitude represents a standard deviation.

You can also **define an error structure from known parameters**. Here, we combine independent measurement noise
with spatial errors. Components are mutually independent, so their variances add.

```{code-cell} ipython3
# Combine independent measurement noise with a spatially correlated error
measurement = gu.ErrorComponent("measurement", magnitude=1.0)
correlation = VariogramModel("spherical", effective_range=100, partial_sill=1)
spatial = gu.ErrorComponent("spatial", magnitude=2.0, correlation=correlation)
errors = gu.ErrorStructure([measurement, spatial])
round(errors.predict_magnitude(), 2), errors.predict_covariance([0.0, 100.0])
```

These components have combined variance `1² + 2² = 5 m²`, giving a magnitude of about **2.24 m**. The correlation
model has unit sill because the component's magnitude supplies its variance.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"
:  image:
:    alt: "Measurement and spatial variances add to 5 m²; spatial covariance falls to zero at 100 m."

# Compare component variances and their combined covariance over distance
figure, axes = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
variances = [
    measurement.predict_magnitude() ** 2,
    spatial.predict_magnitude() ** 2,
    errors.predict_variance(),
]
bars = axes[0].bar(
    ["Measurement", "Spatial", "Combined"],
    variances,
    color=["#3975a7", "#e38b35", "0.25"],
)
axes[0].bar_label(bars, padding=3)
axes[0].set(title="Component variances", ylabel="Error variance (m²)", ylim=(0, 6))

# Show the independent component's contribution at zero distance separately
distances = np.linspace(0.5, 150, 300)
axes[1].plot(
    distances, errors.predict_covariance(distances), color="0.25", label="Combined"
)
axes[1].plot(
    distances,
    4 * spatial.predict_correlation(distances),
    "--",
    color="#e38b35",
    label="Spatial",
)
axes[1].plot(distances, np.zeros_like(distances), color="#3975a7", label="Measurement")
axes[1].scatter(
    [0, 0, 0], [5, 4, 1], color=["0.25", "#e38b35", "#3975a7"], s=20, zorder=3
)
axes[1].axvline(100, color="0.65", linestyle=":")
axes[1].set(
    title="Covariance over distance",
    xlabel="Separation (m)",
    ylabel="Covariance (m²)",
    xlim=(-3, 150),
)
axes[1].legend(frameon=False)
plt.show()
```

The independent component contributes only to an observation's own variance. The spatial component's covariance
falls to zero at **100 m** for this spherical model.

(uncertainty-propagation)=
## Propagating through a spatial operation

{func}`~geoutils.uncertainty.propagate` calculates **how source errors affect a spatial result**.
Pass the spatial method to run, its arguments through `operation_kwargs`, and the source `error_structure`.
Supported methods include raster point resampling, reprojection, point gridding and kriging.

```{code-cell} ipython3
# Average the 3 x 3 cells surrounding two locations and propagate the fitted errors
locations = (np.array([155.0, 205.0]), np.array([165.0, 115.0]))
mean_operator = Mean(neighborhood=GridNeighbours(size=3))
summary = gu.uncertainty.propagate(
    measured.rst.resample_at_points,
    error_structure=estimated_errors,
    operation_kwargs={"points": locations, "method": mean_operator, "as_array": True},
)
np.round(summary.estimate, 2), np.round(summary.std, 2)
```

The {class}`~geoutils.uncertainty.propagation.PropagationSummary` returns three results in the operation's output layout:

| Attribute | Meaning |
| --- | --- |
| `estimate` | Result calculated from the original measurements |
| `mean` | Expected result after adding source errors |
| `std` | Standard deviation of that result |

For a linear calculation with zero-mean errors, `mean` equals `estimate`. Averaging nine independent observations here reduces
the uncertainty from about 2 m to **2/3 m**.

**Spatial correlation limits that reduction.** Compare the measurement + spatial model defined above with an
independent model having the same total source magnitude of 2.24 m:

```{code-cell} ipython3
# Define independent errors with the same magnitude for a fair comparison
independent_component = gu.ErrorComponent(
    "measurement", magnitude=errors.predict_magnitude()
)
independent_errors = gu.ErrorStructure([independent_component])

# Propagate both models through the same averages
comparison = {}
models = {"Independent": independent_errors, "Measurement + spatial": errors}
for name, model in models.items():
    comparison[name] = gu.uncertainty.propagate(
        measured.rst.resample_at_points,
        error_structure=model,
        operation_kwargs={
            "points": locations,
            "method": mean_operator,
            "as_array": True,
        },
    )
(
    np.round(comparison["Independent"].std, 2),
    np.round(comparison["Measurement + spatial"].std, 2),
)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"
:  image:
:    alt: "Uncertainty falls as averaging windows grow, more slowly for spatially correlated errors than independent errors."

# Repeat the same average over larger windows to show the effect of spatial correlation
window_sizes = [1, 3, 5, 9, 15]
window_uncertainties = {"Independent": [], "Measurement + spatial": []}
for size in window_sizes:
    operator = Mean(neighborhood=GridNeighbours(size=size))
    for name, model in models.items():
        result = gu.uncertainty.propagate(
            measured.rst.resample_at_points,
            error_structure=model,
            operation_kwargs={
                "points": locations,
                "method": operator,
                "as_array": True,
            },
        )
        window_uncertainties[name].append(result.std[0])

# Compare output magnitudes at identical source magnitudes and spatial windows
figure, ax = plt.subplots(figsize=(6, 3), layout="constrained")
for name, color in [("Independent", "#3975a7"), ("Measurement + spatial", "#e38b35")]:
    ax.plot(window_sizes, window_uncertainties[name], "o-", color=color, label=name)
ax.set(
    xticks=window_sizes,
    xlabel="Window side length (cells)",
    ylabel="Uncertainty of the mean (m)",
    title="Spatial correlation limits uncertainty reduction",
    ylim=(0, 2.5),
)
ax.legend(frameon=False)
plt.show()
```

Both curves start with the same single-cell uncertainty. Nearby correlated errors partly move together, so
averaging them removes less uncertainty.

### Uncertainty of a statistic

Statistics accept an error model when using a single reducer:

```{code-cell} ipython3
# Return the global mean and its uncertainty from the same selected raster values
mean_value, mean_uncertainty = measured.rst.stats(Mean(), error_structure=estimated_errors)
round(mean_value, 2), np.round(mean_uncertainty.std, 3)
```

For these 4096 independent observations, the mean's uncertainty is about `2 / sqrt(4096) = 0.031 m`.
A spatially correlated model generally gives a larger uncertainty, as in the window comparison above.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"
:  image:
:    alt: "Error bars narrow from one measurement to a local average and the whole-raster mean, with the known 100 m surface shown for comparison."

# Compare a single measurement, a local mean and the whole-raster mean
single_value = measured.rst.resample_at_points(
    points=locations, method="nearest", as_array=True
)[0]
estimates = [single_value, summary.estimate[0], mean_value]
uncertainties = [
    estimated_errors.predict_magnitude(), summary.std[0], mean_uncertainty.std.item()
]
figure, ax = plt.subplots(figsize=(6, 3), layout="constrained")
ax.errorbar(
    estimates, [2, 1, 0], xerr=uncertainties, fmt="o", color="#3975a7", capsize=4,
    label="Estimate ± standard deviation",
)
ax.axvline(100, color="0.25", linestyle="--", label="Known surface")
for row, value, uncertainty in zip([2, 1, 0], estimates, uncertainties):
    ax.annotate(
        f"{uncertainty:.3f} m", (value, row), xytext=(0, 10),
        textcoords="offset points", ha="center",
    )
ax.set(
    yticks=[2, 1, 0], yticklabels=["One measurement", "3 × 3 mean", "Whole-raster mean"],
    xlabel="Measured or averaged value (m)", ylim=(-0.5, 2.6),
    title="Averaging independent measurements reduces uncertainty",
)
ax.legend(loc="lower right", frameon=False, fontsize=8)
plt.show()
```

(uncertainty-advanced)=
## Advanced usage

### Numerical propagation and uncertainty intervals

`method="auto"` uses analytical propagation when the operator provides linear coefficients. Use
`method="numerical"` for repeated random draws (Monte Carlo), including nonlinear calculations.

A median can change which observation supplies the result after errors are added. Here, the three observations
are **10, 11 and 20 m**, each with the fitted independent errors. We store draws to inspect the resulting distribution.

```{code-cell} ipython3
# Use a median to demonstrate numerical propagation for a nonlinear calculation
observations = LocalData(
    values=np.array([10.0, 11.0, 20.0]),
    valid=np.ones(3, dtype=bool),
    source_ids=np.arange(3),
)
median_summary = gu.uncertainty.propagate(
    Median(),
    observations,
    error_structure=estimated_errors,
    method="numerical",
    n_simulations=2000,
    return_samples=True,
    random_state=42,
)
median_summary.estimate, round(median_summary.mean, 2), round(median_summary.std, 2)
```

```{code-cell} ipython3
# Calculate the central 95% interval from the stored output draws
median_summary.interval(0.95)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"
:  image:
:    alt: "Simulated median outputs have a mean of 11.74 m compared with the original 11 m median, with a central 95% interval."

# Show the median's output distribution and its empirical uncertainty interval
lower, upper = median_summary.interval(0.95).iloc[0]
figure, ax = plt.subplots(figsize=(6, 3), layout="constrained")
ax.axvspan(lower, upper, color="#3975a7", alpha=0.12, label="Central 95% interval")
ax.hist(
    median_summary.samples.iloc[:, 0], bins=35, density=True, color="#3975a7", alpha=0.6
)
ax.axvline(
    median_summary.estimate,
    color="0.25",
    label=f"Original median: {median_summary.estimate:.2f} m",
)
ax.axvline(
    median_summary.mean,
    color="#e38b35",
    linestyle="--",
    label=f"Mean after adding errors: {median_summary.mean:.2f} m",
)
ax.set(
    xlabel="Median after adding source errors (m)",
    ylabel="Probability density (1/m)",
    title="Nonlinear calculations can shift the expected result",
)
ax.legend(frameon=False)
plt.show()
```

Zero-mean input errors can shift the expected output of a nonlinear calculation. Increase `n_simulations` for more
stable numerical estimates. `interval()` uses empirical quantiles when draws are stored; analytical intervals
use the normal output law of the supported linear calculation.

For large outputs, select locations with `at=` before storing draws. Set `quantiles=(0.025, 0.975)` to save only
those percentiles in the result, without returning every draw. Exact empirical quantiles still require a temporary
draw buffer, whose size is controlled by `max_sample_bytes`.

(uncertainty-simulation)=
### Simulating random error fields

{meth}`random_field() <RasterBase.random_field>` uses an existing raster or point cloud to define the output
locations. It can generate **synthetic errors with a given magnitude and spatial correlation** for repeated
measurements or numerical experiments.

```{note}
Correlated fields use GSTools' [SRF()](https://geostat-framework.readthedocs.io/projects/gstools/en/stable/api/gstools.field.SRF.html)
or GPyTorch's [MultivariateNormal.rsample()](https://docs.gpytorch.ai/en/stable/distributions.html#gpytorch.distributions.MultivariateNormal.rsample).
GeoUtils adds shared error components and spatially consistent draws across chunks.
```

```{code-cell} ipython3
# Draw independent and correlated fields with the same total error magnitude
independent_field = measured.rst.random_field(
    independent_errors, backend="gstools", random_state=42
)
field = measured.rst.random_field(errors, backend="gstools", random_state=42)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"
:  image:
:    alt: "Simulated errors of equal magnitude vary cell by cell when independent and form patches when spatially correlated."

# Compare independent measurement errors with the simulated correlated field
figure, axes = plt.subplots(1, 2, figsize=(7, 3.2))
axes[0].set_title("Independent errors")
independent_field.rst.plot(ax=axes[0], cmap="RdBu", vmin=-6, vmax=6, cbar_title="Error (m)")
axes[1].set_title("Measurement + spatial errors")
field.rst.plot(ax=axes[1], cmap="RdBu", vmin=-6, vmax=6, cbar_title="Error (m)")
for axis in axes:
    axis.set(xlabel="X (m)", ylabel="Y (m)")
plt.tight_layout()
plt.show()
```

Both models have a **2.24 m magnitude**, but the spatial component creates coherent patches while independent
errors vary from cell to cell. Individual fields need not have exactly the model's standard deviation. Add a
field to `reference` to construct another synthetic elevation measurement; the drawn field contains errors alone.

### Fitting controls

Use `mask=` to select unchanged surfaces when only part of the data provides an error proxy. The magnitude estimate
uses `spread_estimator`; the variogram uses `variogram_estimator`. The standard deviation and Matheron's estimator
suit these normal synthetic errors. NMAD and Dowd's estimator are alternatives that are less sensitive to outliers.

`bins` and `min_count` control predictor groups; `subsample_magnitude` limits the observations used for their spread.
`n_pairs`, `n_lags` and `max_lag` control spatial pair sampling and distance bins. See {ref}`stats-spatial` for
variogram diagnostics, and {ref}`sampling-pairs` for pair selection.

With predictors, the default components are a **heteroscedastic Gaussian** and a **constant spherical** component.
Without predictors, the default is one constant Gaussian component. At most one component can be heteroscedastic.
Use `refit()` to fit new correlation models to the stored diagnostics without sampling the data again.

### Simulation backends and large datasets

{func}`~geoutils.uncertainty.random_field` also accepts an array of coordinates. Use `n_fields=` for several
independent fields, and the same `random_state` and model settings for repeatable draws.

| Backend | How it draws spatial errors | Main control |
| --- | --- | --- |
| `"gstools"` | Shared global random model | Same component seeds across chunks |
| `"gpytorch"` (default) | Approximation on an inducing grid for spatial inputs | `gpytorch_inducing_points=` (256 by default); `None` requests an exact eager draw |

The inducing approximation can change the represented covariance; choose its size according to the required
accuracy and available memory. Chunked fields share their random model across chunks, preserving correlation
across chunk boundaries.

Magnitude prediction, estimation and simulation support Dask and multiprocessing. Marginal propagation can
remain lazy on supported Dask inputs; selected spatial samples and quantiles currently require eager inputs.
See {ref}`scalability-support` for execution limits and {ref}`stats-kriging` for predictions from measured values.
