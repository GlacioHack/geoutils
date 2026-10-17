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
(core-operators)=
# Operators and neighborhoods

GeoUtils defines **operators and neighborhoods** shared by many raster and point cloud calculations.

An **operator** is either an **interpolator** or a **reducer** and defines the calculation, while a **neighborhood** selects the nearby observations it uses. This separation
allows the use of **consistent methods** across operations, including reducers in {ref}`filters` or {ref}`stats`,
and both operator types for resampling in {ref}`transformations` on points or rasters (e.g., reprojection, gridding).

GeoUtils sets **convenient default behavior matching GDAL/PDAL**,
with a broader choice of methods (e.g., kriging, inverse-distance-weighting), neighborhood definitions, nodata
propagation schemes, and numerical backends (Rasterio/GDAL, SciPy or Numba) for efficiency.

For more **advanced use**, GeoUtils supports customizable operators, allowing users to vary method or backend without having to
reinvent the GIS machinery or chunked execution specific to an operation (i.e. maintains Dask support).

Below, we first give a broad description of {ref}`core-interpolation-reduction`,
{ref}`core-neighborhoods`, {ref}`core-coverage` and
{ref}`core-nodata`.
Then, we delve into how to {ref}`core-custom-operators`.

(core-interpolation-reduction)=
## Interpolators and reducers

An **interpolator predicts a continuous value at a requested coordinate**, usually using the positions of nearby observations.
A **reducer summarizes values within a neighborhood**, for example by their mean or median.


```{eval-rst}
.. plot:: diagrams/diagram_operators.py
    :width: 100%
    :alt: Cubic interpolation and a mean of five observations, with both results shown at the central target.
```

```{note}
When applied to **resampling** in transformations, the choice of using an **interpolator** or a **reducer** relates to the support interpretation of a raster or point dataset.
In other words, to the question: **Does this data aim to represent an average value over a surrounding (pixel/footprint), or the exact value at a point (pixel center, point coordinates)?**

See {ref}`pixel-interpretation` for details on this interpretation in raster metadata.
```

Choose the calculation with `resampling=` for transformations and gap filling, `method=` for filters,
or `statistics=` for statistics. Where available, `backend=` selects the implementation.
Extra settings use `resampling_kwargs=` for transformations or `method_kwargs=` for filters.

Compare {ref}`built-in interpolators <transformations-interpolators>` and {ref}`reducers <transformations-reducers>`
with their available backends on the {ref}`transformations` page.

(core-neighborhoods)=
## Grid and point neighborhoods

A **grid neighborhood selects source pixels**, while a **point neighborhood selects nearby observations**:

- For raster filtering, resampling and reprojection, use `window=` and `window_shape=`. Sizes are in source pixels,
  with square or circular windows.
- For point filtering, `radius=` limits the X/Y distance in CRS units and `max_points=` limits the number of nearest points.
  Gridding uses `radius=` for a search distance in output pixels.

```{eval-rst}
.. plot:: diagrams/diagram_neighbours.py
    :width: 100%
    :alt: A bounded search on points and a circular window on a raster grid.
```

Here, `max_points=4` caps the point neighborhood at four observations. Grid interpolation uses the cells required by
its method. A finite radius bounds the surrounding data needed for chunked point calculations;
see {ref}`scalability-support` for the requirements.

(core-coverage)=
## Cell coverage

Raster cells cover an area, so a window or polygon can intersect only part of a cell. `coverage=` selects
their contribution to raster reductions, filters, polygon statistics and rasterization:

| Coverage | Contribution of an edge cell |
|---|---|
| `"center"` | Include its whole value when its center falls inside the area |
| `"all_touched"` | Include its whole value when any part intersects the area |
| `"fractional"` | Weight its value by the fraction of the cell covered by the window or polygon |

```{eval-rst}
.. plot:: diagrams/diagram_coverage.py
    :width: 100%
    :alt: Center, all touched and fractional coverage for the same circle, with every fractional percentage labeled.
```

A cell with 25% coverage has a geometric weight of 0.25. See {ref}`core-weighting` for fractional means and sums.
Reprojection uses the output cell's footprint in the source grid unless an explicit window is supplied.
Polygon examples are in {ref}`transformations-fractional` and {ref}`stats-fractional`.

(core-nodata)=
## Nodata propagation

GeoUtils treats masked or non-finite observations as missing. The nodata rule determines whether to omit them from
the calculation or make the output missing too.

| Rule | Behavior |
|---|---|
| `"ignore"` | Calculate from valid observations, if enough remain |
| `"propagate"` | Return NaN when a missing observation contributes to the result |
| `"nearest"` | Mask an interpolated output when its nearest source observation is missing |
| Non-negative integer | Expand the raster nodata mask by this many source pixels, then mask the outputs it covers |
| `"half_order_up"` / `"half_order_down"` | Derive that distance from half the interpolation order, rounded up or down |

```{eval-rst}
.. plot:: diagrams/diagram_nodata.py
    :width: 100%
    :alt: The same five observations with one missing value, giving a valid mean when ignored and NaN when propagated.
        With quarter-pixel-shifted outputs and a five-cell window, propagation masks every output, nearest masks
        only the middle output, and a one-pixel spread masks the middle three outputs.
```

Use `nodata_propagation=` for reprojection and point filters, or `nodata_handling=` for point resampling,
gridding and raster reducer filters. `"nearest"` applies to interpolation. For raster edges, `boundless=True`
allows partial windows; `False` requires complete windows. These choices apply to the available neighborhood
without changing its size.

For raster point resampling, pass a distance choice with `nodata_handling=`, such as `nodata_handling=1` above.

(core-operator-uncertainty)=
## Weights and uncertainty

Reducers and interpolators share a composable interface for weights and source errors. For example, Mean combines
sample and fractional coverage weights, while Mean and InverseDistance can both adjust their coefficients using
an `error_structure=` model. The same operators can be passed through a spatial method to
{func}`~geoutils.uncertainty.propagate` to calculate output uncertainty, analytically for linear operators or
numerically for nonlinear ones.

See {ref}`core-weighting` for worked examples on real data across sampling and reprojection, and
{ref}`uncertainty` for defining and estimating the error structure itself.

(core-custom-operators)=
## Customize operators and neighborhoods

For more advanced use, GeoUtils also exposes **customizable** {class}`~geoutils.operators.Interpolator`,
{class}`~geoutils.operators.Reducer`, {class}`~geoutils.operators.GridNeighbours` and
{class}`~geoutils.operators.PointNeighbours` objects. Those allow users to easily extend these operators with their own
methods/backend while reusing the geospatial machinery and chunked execution (Dask or multiprocessing) of an operation
already implemented in GeoUtils.

### Configuring methods and neighborhoods

We create a small raster and compare two named calculations at the same locations:

```{code-cell} ipython3
import numpy as np
from affine import Affine

import geoutils as gu

# Create a small elevation grid with 10 m cells in a projected CRS
elevations = np.arange(25, dtype=float).reshape(5, 5)
raster = gu.Raster.from_array(elevations, Affine(10, 0, 0, 0, -10, 50), 32631, nodata=-9999)
locations = (np.array([20.0, 30.0]), np.array([30.0, 20.0]))

# Predict elevations at the locations, then average a window around each one
interpolated = raster.resample_at_points(locations, resampling="linear", backend="scipy", as_array=True)
averaged = raster.resample_at_points(locations, resampling="mean", backend="scipy", window=3, as_array=True)
interpolated, averaged
```

Operator objects group the method and its neighborhood settings. GridNeighbours supports square and circular
windows, or explicit `(row, column)` offsets from the containing source cell:

```{code-cell} ipython3
from geoutils.operators import GridNeighbours, PointNeighbours
from geoutils.operators.interpolator import InverseDistance
from geoutils.operators.reducer import Mean

window = GridNeighbours(size=3)
square = GridNeighbours(size=5)
circle = GridNeighbours(size=5, shape="circular")
cross = GridNeighbours(offsets=((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)))
```

PointNeighbours combines a point count and radius. Distances use **horizontal X/Y coordinates in CRS units**;
reproject geographic coordinates to a metric CRS for radii in metres. In gridding, this differs from the
output pixels used by `radius` in grid():

```{code-cell} ipython3
# Create point measurements from the same grid
points = raster.to_pointcloud()

# Grid values using at most eight measurements within 25 m of each output cell
nearby = PointNeighbours(max_points=8, radius=25)
gridded = points.grid(ref=raster, resampling=InverseDistance(neighborhood=nearby))
gridded.shape
```

### Combining coverage and nodata rules

For fractional windows, the shape is centered on the **requested coordinate**, so moving a point within a cell
changes the coverage weights. A reducer must support those weights to use fractional coverage.

```{code-cell} ipython3
# Average the parts of cells covered by a circle centered on each requested point
circle = GridNeighbours(size=3, shape="circular", coverage="fractional")
covered_mean = raster.resample_at_points(locations, resampling=Mean(neighborhood=circle), as_array=True)
covered_mean
```

An operator's `default_nodata_propagation` sets its own default when the operation allows it. For linear operators,
a missing observation with a coefficient of zero does not affect the result. The call can override that default:

```{code-cell} ipython3
# Create a missing observation in the window used by the first location
with_gap = raster.copy()
with_gap.data[1, 1] = np.ma.masked

# Compare an average of valid cells with propagation of the missing value
ignored = with_gap.resample_at_points(locations, resampling=Mean(neighborhood=window), nodata_handling="ignore", as_array=True)
propagated = with_gap.resample_at_points(locations, resampling=Mean(neighborhood=window), nodata_handling="propagate", as_array=True)
ignored, propagated
```

Raster interpolation also accepts a non-negative integer distance or `"half_order_up"` / `"half_order_down"` in
`nodata_handling`. These options spread the missing-data mask by a distance in source pixels; the latter two derive
that distance from interpolation order. They control the output mask separately from the prediction itself.

Custom operators receive a {class}`~geoutils.operators.LocalData` object for each output. It contains
one-dimensional `values`, their validity and source IDs, and, when needed, coordinates, distances and weights.
GeoUtils selects the neighborhood and applies the nodata rule before calling your calculation.

### Custom reducers

Implement `reduce()` to combine the selected values. Here we remove the lowest and highest values before averaging,
which can limit the influence of an isolated extreme observation:

```{code-cell} ipython3
class TrimmedMean(gu.operators.Reducer):
    """Average nearby values after removing their lowest and highest observations."""

    minimum_inputs = 3

    def reduce(self, data):
        """Sort the valid observations and average the interior values."""
        ordered = np.sort(data.values)
        return float(np.mean(ordered[1:-1]))


# Apply the custom reducer to the same raster windows
trimmed = raster.resample_at_points(locations, resampling=TrimmedMean(neighborhood=window), as_array=True)
trimmed
```

Declare `accepts_sample_weights` or `accepts_support_weights` only when the implementation uses those weights.
GeoUtils raises an error for unsupported weights instead of silently discarding them.

### Custom interpolators and linear coefficients

Implement `predict()` for a custom interpolator. If the result can be written as a **weighted sum**, implement
`coefficients()` instead and return {class}`~geoutils.operators.LinearCoefficients`. GeoUtils can then calculate
the prediction and reuse its coefficients for analytical uncertainty propagation.

```{code-cell} ipython3
class DistanceMean(gu.operators.Interpolator):
    """Predict a value by giving closer observations more weight."""

    def coefficients(self, data):
        """Normalize inverse distances, giving coincident observations precedence."""

        # Avoid division by zero when the requested point matches a source location
        coincident = data.distances == 0
        if np.any(coincident):
            weights = coincident.astype(float)
        else:
            weights = 1 / data.distances

        # Account for observation covariance when an error model is supplied
        weights = data.precision_weights(weights)
        return gu.operators.LinearCoefficients(weights)


# Use the custom interpolation through the public raster API
predicted = raster.resample_at_points(locations, resampling=DistanceMean(neighborhood=GridNeighbours(size=3)), backend="scipy", as_array=True)
predicted
```

The same `coefficients()` contract applies to linear reducers. It requires one coefficient per selected value and
allows a constant offset. See {ref}`core-weighting` for how these coefficients relate to weights and uncertainty.
