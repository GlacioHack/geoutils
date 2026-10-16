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
(core-numerical-interface)=
# Numerical interfacing

GeoUtils links to existing **numerical interfaces** to facilitate raster and point cloud arithmetic.

Xarray and GeoPandas provide **Python operators** and **NumPy function support** directly on their objects.
GeoUtils implements its own interfaces for {class}`~geoutils.Raster` and {class}`~geoutils.PointCloud`, handling 
**NumPy's masked arrays** so calculations respect invalid pixels.

The following sections demonstrate how to perform basic arithmetic, comparing {ref}`Python operators <core-py-ops>` and 
{ref}`NumPy functions and reductions <core-array-funcs>` across GeoUtils objects,
{class}`~xarray.DataArray` and {class}`~pandas.Series`.

```{note}
The examples use the same table format for masked arrays and arrays with NaNs, displaying missing values as `—`
to make the results easier to compare.
```

```{code-cell} ipython3
:tags: [remove-cell]

import geoutils as gu
import numpy as np
import pandas as pd
import xarray as xr


def _compact_values_html(value):
    """Show the example values in small HTML tables, with a dash for missing data."""

    # Display matching results side by side without their object metadata
    if isinstance(value, tuple):
        tables = "".join(_compact_values_html(item) for item in value)
        return '<div style="display:flex; gap:1rem; flex-wrap:wrap;">' + tables + "</div>"

    # Extract numeric values while preserving the Raster validity mask
    if isinstance(value, (gu.Raster, gu.PointCloud)):
        values = value.data
    elif isinstance(value, pd.DataFrame):
        values = value.select_dtypes(include="number").to_numpy()
    elif isinstance(value, (xr.DataArray, pd.Series)):
        values = value.to_numpy()
    else:
        values = value

    array = np.ma.asarray(values).squeeze()
    data = np.asarray(array.data, dtype=object)
    data = np.where(np.ma.getmaskarray(array), np.nan, data)

    # Label each result by its type and omit row and column numbering
    table = pd.DataFrame(np.atleast_2d(data)).style
    caption = "Scalar" if np.isscalar(value) else type(value).__name__
    table = table.format(precision=3, na_rep="—").hide(axis="index").hide(axis="columns")
    table = table.set_caption(caption).set_table_styles(
        [{"selector": "", "props": [("width", "auto"), ("margin", "0")]}]
    )
    return table.to_html()


html_formatter = get_ipython().display_formatter.formatters["text/html"]
for value_type in (gu.Raster, gu.PointCloud, xr.DataArray, pd.DataFrame, pd.Series, np.ndarray, tuple):
    html_formatter.for_type(value_type, _compact_values_html)
```

(core-py-ops)=
## Python operators

### Arithmetic

Arithmetic operators ({func}`+<operator.add>`, {func}`-<operator.sub>`, {func}`/<operator.truediv>`, {func}`//<operator.floordiv>`, {func}`*<operator.mul>`,
{func}`**<operator.pow>`, {func}`%<operator.mod>`) can be used on a {class}`~geoutils.Raster` in combination with any other {class}`~geoutils.Raster`,
{class}`~numpy.ndarray` or number.

With another Raster, its shape, {attr}`~RasterBase.crs` and {attr}`~RasterBase.transform` must match.
With a NumPy array, its shape must match. The result is a Raster, with masked values propagated.
Dtypes follow [NumPy's promotion rules](https://numpy.org/doc/stable/reference/arrays.promotion.html).

The same is true for a {class}`~geoutils.PointCloud`. However, different object types cannot perform arithmetic directly with each other. 
To do so, first reconcile them to the same object type (i.e. spatial support) either at high-level through {ref}`sampling-cosample` or directly with {ref}`transformations`.

We create the same small raster with both a {class}`~geoutils.Raster` and {class}`~xarray.DataArray`, using `float32` so Xarray can
represent missing pixels as NaNs:

```{code-cell} ipython3
import geoutils as gu
import rasterio as rio
import pyproj
import numpy as np

# Create a random 3 x 3 masked array
np.random.seed(42)
arr = np.random.randint(0, 255, size=(3, 3), dtype="uint8").astype("float32")
mask = np.random.randint(0, 2, size=(3, 3), dtype="bool")
ma = np.ma.masked_array(data=arr, mask=mask)

# Create an example raster
rast = gu.Raster.from_array(
       data = ma,
       transform = rio.transform.from_bounds(0, 0, 1, 1, 3, 3),
       crs = pyproj.CRS.from_epsg(4326),
       nodata = 255
    )

# Convert masked pixels to NaNs in Xarray
da = rast.to_xarray()
rast, da
```

Arithmetic with a number, an array, or another raster uses the same syntax:

```{code-cell} ipython3
# Arithmetic with a number
rast + 1, da + 1
```

```{code-cell} ipython3
# Arithmetic with an array
rast / arr, da / arr
```

```{code-cell} ipython3
# Arithmetic with a raster
rast - (rast**0.5), da - (da**0.5)
```

```{caution}
:class: margin

{class}`~xarray.DataArray` [aligns index labels](https://docs.xarray.dev/en/stable/user-guide/computation.html#automatic-alignment),
and {class}`~pandas.Series` aligns indices, neither checks the CRS.
Use {ref}`core-match-ref` to match georeferencing first.
```

For a {class}`~geoutils.PointCloud`, arithmetic applies to its active {attr}`~PointCloudBase.data_name` values.
The same values are stored in a {class}`~xarray.DataArray` or a numeric {class}`~pandas.Series`
selected from a {class}`~geopandas.GeoDataFrame`:

```{code-cell} ipython3
# Create three point heights and convert to Xarray and GeoPandas
pc = gu.PointCloud.from_xyz([0, 1, 2], [0, 0, 0], [100.0, 110.0, 120.0], crs=4326, data_name="height")
points = pc.to_xarray()
frame = pc.gdf
height = frame["height"]

# Arithmetic with the same number in each interface
pc + 1, points + 1, height + 1
```

Two {class}`~geoutils.PointCloud` objects must have matching X/Y coordinates and CRS, and an array the same shape.

### Logical comparisons

Logical comparison operators ({func}`==<operator.eq>`, {func}` != <operator.ne>`, {func}`>=<operator.ge>`, {func}`><operator.gt>`, {func}`<=<operator.le>`,
{func}`<<operator.lt>`) can be used with the same inputs as arithmetic. They return a boolean {class}`~geoutils.Raster` or  {class}`~geoutils.PointCloud`,
or a boolean {class}`~xarray.DataArray` or {class}`~pandas.Series`.

```{note}
:class: margin

Notice that {class}`Rasters <geoutils.Raster>` can have masked boolean (`True`/`False`) values,
while boolean {class}`DataArrays <xarray.DataArray>` cannot. When we use them to index or assign values,
both use the booleans directly, to ensure no difference on behaviour.
```

```{code-cell} ipython3
# Logical comparison with a number
mask = rast > 100
da_mask = da > 100
mask, da_mask
```

```{code-cell} ipython3
# Compare the same point heights
pc > 100, points > 100, height > 100
```

### Combining boolean conditions

Logical bitwise operators ({func}`~ <operator.invert>`, {func}`& <operator.and_>`, {func}`| <operator.or_>`, {func}`^ <operator.xor>`)
can combine boolean Rasters, PointClouds, arrays, {class}`~xarray.DataArray`, or {class}`~pandas.Series`:

```{code-cell} ipython3
# Logical bitwise operation between masks
mask = (rast > 100) & ((rast % 2) == 0)
da_mask = (da > 100) & ((da % 2) == 0)
mask, da_mask
```

```{code-cell} ipython3
# Combine the same conditions on point heights
pc_mask = (pc > 100) & ((pc % 2) == 0)
point_mask = (points > 100) & ((points % 2) == 0)
height_mask = (height > 100) & ((height % 2) == 0)
pc_mask, point_mask, height_mask
```

(py-ops-indexing)=
### Boolean indexing and assignment

Indexing a {class}`~geoutils.Raster` with a matching boolean {class}`~geoutils.Raster` or NumPy array returns a flattened masked array.
For a {class}`~xarray.DataArray`, [where()](https://docs.xarray.dev/en/stable/user-guide/indexing.html#masking-with-where)
preserves the grid, replacing unselected pixels with NaNs:

```{code-cell} ipython3
# Indexing the raster with the previous mask
rast[mask], da.where(da_mask)
```

For points, we can select rows in all three interfaces:

```{code-cell} ipython3
# Select the point heights matching the previous conditions
pc[pc_mask], points.where(point_mask, drop=True), frame.loc[height_mask]
```

We can also use the same masks to assign zeros:

```{code-cell} ipython3
# Copy the PointCloud and Series so later examples can use the original values
pc_assigned = pc.copy()
height_assigned = height.copy()

# Assign zero where each mask is True
pc_assigned[pc_mask] = 0
points_assigned = points.where(~point_mask, other=0)
height_assigned.loc[height_mask] = 0

# Show the modified values
pc_assigned, points_assigned, height_assigned
```

(core-array-funcs)=
## NumPy functions and reductions

### Universal functions

Universal functions operate element by element, such as {obj}`~numpy.add`, {obj}`~numpy.absolute`,
{obj}`~numpy.isnan`, or {obj}`~numpy.sin`. For {class}`~geoutils.Raster` or {class}`~geoutils.PointCloud` inputs, they can return one or two
objects of the same type. The same shape and georeferencing checks apply as for arithmetic.

Below, we reuse the examples from {ref}`core-py-ops`. The function calls are unchanged with a {class}`~xarray.DataArray`:

```{code-cell} ipython3
# Universal function with a single input and output
np.sin(rast), np.sin(da)
```

```{code-cell} ipython3
# Universal function with two inputs and a single output
np.add(arr, rast), np.add(arr, da)
```

```{code-cell} ipython3
# Universal function with a single input and two outputs (fractional and integer parts)
# Divide values by 10 to give the pixels fractional parts
rast_fraction, rast_integer = np.modf(rast / 10)
da_fraction, da_integer = np.modf(da / 10)

# Compare fractional parts
rast_fraction, da_fraction
```

```{code-cell} ipython3
# Compare integer parts
rast_integer, da_integer
```

NumPy logical comparisons also return boolean results:

```{code-cell} ipython3
# Compare the raster with the same noisy values in both interfaces
noise = np.random.normal(size=np.shape(arr))
np.greater(rast, rast + noise), np.greater(da, da + noise)
```

For points, the same function acts on the {class}`~geoutils.PointCloud`, {class}`~xarray.DataArray`, or {class}`~pandas.Series`:

```{code-cell} ipython3
# Apply the same universal function to point heights
np.sin(pc), np.sin(points), np.sin(height)
```

### Array functions

GeoUtils also supports NumPy array functions, including reductions such as {func}`~numpy.mean`,
{func}`~numpy.count_nonzero` and {func}`~numpy.nanmax`. Functions preserving the original shape return a {class}`~geoutils.Raster`
or {class}`~geoutils.PointCloud`, others return scalars or arrays.

With a {class}`~xarray.DataArray`, use native methods and dimension names instead of axis numbers:

```{code-cell} ipython3
# Traditional mathematical function
np.max(rast), da.max(skipna=True)
```

```{code-cell} ipython3
# Specify an axis for reduction (x is axis 1 of the Raster array)
np.count_nonzero(rast, axis=1), (da.notnull() & (da != 0)).sum(dim="x")
```

The {meth}`~xarray.DataArray.notnull` condition excludes NaNs from the nonzero count.
For points, reductions follow the same pattern:

```{code-cell} ipython3
# Compute the maximum point height in each interface
np.max(pc), points.max(skipna=True), height.max(skipna=True)
```

### Respecting missing values

To use numerical functions while respecting masked pixels, apply a supported NumPy function directly to
the Raster, or a masked-array function to {attr}`~RasterBase.data`. With a {class}`~xarray.DataArray`, use `skipna=True`:

```{note}
:class: margin

Masked-array NumPy functions such as `np.ma.median` are not recognized when applied directly to a Raster, 
because no interface is yet exposed in NumPy. Apply them to {attr}`~RasterBase.data` instead.
```

```{code-cell} ipython3
# Numpy core function applied to the raster
np.median(rast), da.median(skipna=True)
```

For the underlying NumPy array of a {class}`~xarray.DataArray`, NaN functions serve the same purpose:

```{code-cell} ipython3
# Numpy NaN function applied to the raster
np.nanmedian(rast), np.nanmedian(da.values)
```

```{code-cell} ipython3
# Masked-array function on the data
np.ma.median(rast.data), np.nanmedian(da.values)
```

For a {class}`~geoutils.PointCloud` we can use NaN functions, while {class}`~pandas.Series` reductions accept `skipna=True`:

```{code-cell} ipython3
# Set the middle height to NaN and convert the same points to each interface
pc_missing = pc.copy(new_array=np.array([100.0, np.nan, 120.0]))
points_missing = pc_missing.to_xarray()
height_missing = pc_missing.gdf["height"]

# Compute the median point height while ignoring missing values
np.nanmedian(pc_missing), points_missing.median(skipna=True), height_missing.median(skipna=True)
```

More generally to compute global or grouped statistics while respecting nodata, sampling multiple datasets at co-located areas, 
while maintaining laziness and chunked execution on large datasets, use the **{ref}`stats` functionalities**.
