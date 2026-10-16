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
(core-weighting)=
# Weights and uncertainty

The {ref}`operators <core-operators>` used for spatial calculations share support for **weights and uncertainty**.
You can reuse a reducer or interpolator across operations, combine its geometric weights with source precision,
and propagate an error model through the same calculation.

This page demonstrates that composability on the **ASTER DEM of Exploradores glacier**. See {ref}`uncertainty`
for the definition, estimation and simulation of error structures.

## Reuse a weighted reducer

A weight controls how much an observation contributes. Its meaning depends on the calculation:

| Weight | Where it comes from | Example |
|---|---|---|
| Geometric weight | Covered cell area or distance to the requested location | Fractional window coverage or inverse-distance interpolation |
| Sample weight | Importance assigned to an observation | Measurements with different representativeness |
| Precision weight | Source error covariance | A mean giving more weight to measurements with smaller independent errors |

We open the example DEM through its Xarray accessor and crop a 24 × 24 cell area for the calculations below:

```{code-cell} ipython3
import pandas as pd

import geoutils as gu
from geoutils.operators import GridNeighbours
from geoutils.operators.interpolator import InverseDistance
from geoutils.operators.reducer import Mean

# Open the DEM and select a small area around its center
dem = gu.open_raster(gu.examples.get_path("exploradores_aster_dem"))
height, width = dem.rst.shape
row, column = height // 2, width // 2
dem = dem.rst.icrop((column - 12, row - 12, column + 12, row + 12))

# Choose two locations between cell centers so their coverage and distances differ
locations = (
    dem.x.values[[8, 16]] + dem.rst.res[0] / 4,
    dem.y.values[[8, 16]] + dem.rst.res[1] / 4,
)
```

{class}`~geoutils.operators.reducer.Mean` supports fractional coverage: each cell contributes in proportion to
the area intersected by the window. We use the **same reducer** to average elevations at the selected locations
and on a coarser grid:

```{code-cell} ipython3
# Configure an area average over a circular window three source pixels wide
circle = GridNeighbours(size=3, shape="circular", coverage="fractional")
area_mean = Mean(neighborhood=circle)

# Apply the reducer to point sampling and raster reprojection
sampled = dem.rst.resample_at_points(locations, resampling=area_mean, as_array=True)
coarse = dem.rst.reproject(res=60, resampling=area_mean)
sampled, coarse.rst.shape
```

Both calls use the reducer's coverage and normalization rules; the requested output coordinates determine
which cells contribute. The same reducer can also be used in {ref}`filters` and {ref}`stats`.

For reducers accepting both sample and coverage weights, these multiply. A **mean** normalizes the weights,
while a **sum** uses them directly. For elevation we want an average; summing fractions of cell totals is appropriate
for quantities such as population counts. Integrating a density also requires cell area in suitable units.
Weighted quantiles require an explicit choice, such as `Median(weighted=True)`. Operators raise an error when
they cannot use supplied weights.

(core-error-weighting)=
## Use source errors across operators

An {class}`~geoutils.ErrorStructure` can adjust the weights used by both a **reducer** such as Mean and an
**interpolator** such as InverseDistance. For independent errors, smaller error variances give larger precision
weights. With correlated errors, observations are considered together, and the resulting coefficients can be negative.

For this example, we **assume** elevation errors of 10 m with spherical correlation over 300 m. These values
illustrate the calculation; they are not an uncertainty estimate for this DEM. To fit a model from measurement
differences and predictors, see {ref}`uncertainty-estimation`.

```{code-cell} ipython3
from geoutils.stats.variography import VariogramModel

# Define the assumed source errors in the DEM's elevation and projected coordinate units
correlation = VariogramModel("spherical", effective_range=300, partial_sill=1)
errors = gu.ErrorStructure([gu.ErrorComponent("measurement", magnitude=10.0, correlation=correlation)])

# Give both operators the same 3 x 3 neighborhood
window = GridNeighbours(size=3)
operators = {"Mean": Mean(neighborhood=window), "Inverse distance": InverseDistance(neighborhood=window)}

# Compare the ordinary estimates with estimates using source error covariance
predictions = {}
for name, operator in operators.items():
    predictions[(name, "Ordinary")] = dem.rst.resample_at_points(locations, resampling=operator, as_array=True)
    predictions[(name, "With source errors")] = dem.rst.resample_at_points(
        locations, resampling=operator, error_structure=errors, as_array=True
    )
pd.DataFrame(predictions, index=["Location 1", "Location 2"]).round(2)
```

Supplying `error_structure=` uses the model for fitting where the operator supports it and returns the usual
spatial output. Geometric weights still describe coverage or distance; the error model supplies the covariance
used to adjust their contribution. The same argument is available in raster reprojection and point gridding.

(core-linear-operators)=
## Propagate through the same calculation

Use {func}`~geoutils.uncertainty.propagate` with a **bound spatial method** and its arguments to calculate output
uncertainty. We reuse the operators, locations and error structure from above:

```{code-cell} ipython3
# Calculate each estimate and its standard deviation using the same spatial calls
summaries = {}
for name, operator in operators.items():
    summaries[name] = gu.uncertainty.propagate(
        dem.rst.resample_at_points,
        error_structure=errors,
        operation_kwargs={"points": locations, "method": operator, "as_array": True},
    )

# Display elevations and their propagated uncertainty in metres
propagated = {}
for name, summary in summaries.items():
    propagated[(name, "Elevation (m)")] = summary.estimate
    propagated[(name, "Uncertainty (m)")] = summary.std
pd.DataFrame(propagated, index=["Location 1", "Location 2"]).round(2)
```

The estimates match the **With source errors** columns above. Both operators expose linear coefficients, so
propagation uses those coefficients analytically. Here, **linear** means a weighted sum with coefficients fixed
for the supplied geometry and error model; it applies to both reductions and interpolations.
The inverse distance result has larger uncertainty here: prioritizing proximity can produce coefficients
that amplify correlated source errors.

The same interface returns raster estimates and uncertainties for reprojection:

```{code-cell} ipython3
# Reuse the area average to calculate elevations and uncertainty on the coarser grid
coarse_summary = gu.uncertainty.propagate(
    dem.rst.reproject,
    error_structure=errors,
    operation_kwargs={"res": 60, "resampling": area_mean},
)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

from matplotlib import pyplot as plt

# Compare the source elevations, the weighted area average and its uncertainty
plt.rcParams["figure.dpi"] = 600
plt.rcParams["font.size"] = 9
figure, axes = plt.subplots(1, 3, figsize=(9, 3))
elevation_limits = {"vmin": float(dem.min()), "vmax": float(dem.max())}
dem.rst.plot(ax=axes[0], cmap="terrain", cbar_title="Elevation (m)", **elevation_limits)
coarse_summary.estimate.rst.plot(ax=axes[1], cmap="terrain", cbar_title="Elevation (m)", **elevation_limits)
coarse_summary.std.rst.plot(ax=axes[2], cmap="viridis", vmin=0, vmax=10, cbar_title="Uncertainty (m)")

# Shorten projected coordinate labels so the panels remain readable
for axis, title in zip(axes, ["Source DEM (30 m)", "Weighted mean (60 m)", "Propagated uncertainty"]):
    axis.set_title(title)
    axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useOffset=False)
plt.tight_layout()
```

Positive spatial correlation limits the reduction in uncertainty from averaging nearby cells. The plotted
standard deviation describes the assumed **measurement errors propagated through the operator**, rather than
the variation in terrain elevation within each window. At the crop's edges, partial windows contain fewer cells
and give larger uncertainty.

For nonlinear operators such as a median, the same interface uses numerical propagation, and `summary.mean`
can differ from `summary.estimate`. See {ref}`uncertainty-propagation` for numerical examples and execution options.
Statistics with a single reducer provide this composition directly through `stats(Mean(), error_structure=errors)`,
returning the statistic and its propagation summary.

These calls explicitly propagate a supplied source model through one operation. Ordinary arithmetic does not
automatically track uncertainty through a sequence of operations. Custom operators can participate by providing
linear coefficients or a calculation for numerical draws, as shown in {ref}`core-custom-operators`.
