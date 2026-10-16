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
(core-composition)=

# Data objects and accessors

GeoUtils supports several **data structures** to represent rasters, vectors and point clouds.

First, it extends Xarray and GeoPandas objects with all of its operations by implementing **[accessors](https://docs.xarray.dev/en/latest/internals/extending-xarray.html)**, 
facilitating **interoperability** with other tools accessing these data structures.

Second, it also implements dedicated {class}`~geoutils.Raster`, {class}`~geoutils.Vector` and
{class}`~geoutils.PointCloud` **objects** aimed at **ease-of-use** for Python beginners, to 
manipulate geospatial data without knowing how labeled N-D arrays or dataframes work.

You can choose the structure that suits your workflow. **We recommend using the accessors** for more seamless use 
with the rest of the Python ecosystem, in particular for Dask support.

As the internal design of GeoUtils separates operations from data structures, it **makes it easy to add accessors** 
to emerging structures such as [GeoPolars](https://geopolars.org/latest/) and [Xvec](https://xvec.readthedocs.io/en/stable/) as they become available.

Below, we describe different aspects of these data structures:
- The {ref}`core-structure-summary` section summarizes all supported data structures, their advantages and drawbacks,
- The {ref}`core-datasets` section explains the behaviour of Xarray Dataset accessors and when to prefer it to the DataArray accessors,
- The {ref}`core-point-accessors` section compares Xarray and GeoPandas point accessors, and when to prefer one from the other,
- The {ref}`core-geoutils-objects` section summarizes the behaviour of GeoUtils own objects,
- Finally, the {ref}`core-inheritance` section explains inheritance within GeoUtils and in other packages.

(core-structure-summary)=
## Summary of data structures

GeoUtils currently implements the following **objects** and **accessors**:

| Data type            | GeoUtils accessor | GeoUtils object |
|----------------------|---|---|
| Raster               | {class}`rst <geoutils.DataArrayRasterAccessor>` on {class}`xarray.DataArray` | {class}`~geoutils.Raster` |
| Several rasters      | {class}`rst <geoutils.DatasetRasterAccessor>` on {class}`xarray.Dataset` | — |
| Vector               | {class}`vct <geoutils.GeoDataFrameVectorAccessor>` on {class}`geopandas.GeoDataFrame` | {class}`~geoutils.Vector` |
| Point cloud          | {class}`pc <geoutils.DataArrayPointCloudAccessor>` on {class}`xarray.DataArray`<br>{class}`pc <geoutils.GeoDataFramePointCloudAccessor>` on {class}`geopandas.GeoDataFrame` | {class}`~geoutils.PointCloud` |
| Several point clouds | {class}`pc <geoutils.DatasetPointCloudAccessor>` on {class}`xarray.Dataset` | — |

The advantages and drawbacks between these structures are mainly centered around scalable execution: accessors support **Dask**, while GeoUtils objects only support file-to-file **multiprocessing**. 
Both include support for **chunked execution** on larger-than-memory datasets, but only Dask is **lazy**. See **Scalability's {ref}`scalability-concept` page** for more details on these concepts.

Other than scalability, differences are small:
- **Rasters:** Xarray accessor requires floating NumPy arrays with NaNs, while a {class}`~geoutils.Raster` uses more memory-efficient NumPy masked arrays for integer data, and checks georeferencing during arithmetic.
- **Vectors:** Few differences, mainly that a {class}`~geoutils.Vector`'s geometric operations always return {class}`~geoutils.Vector` objects (does not differentiate series and dataframes).
- **Point clouds:** Few differences, mainly that a {class}`~geoutils.PointCloud` supports implicit arithmetic tied to a main data column/variable, while the two point accessors require an explicit column/variable. Differences between the accessors are discussed in {ref}`core-point-accessors`.

The major practical difference concerns the creation of these objects: use {func}`~geoutils.open_raster`, {func}`~geoutils.open_vector` or
{func}`~geoutils.open_pointcloud` to open Xarray or GeoPandas objects with accessors. For GeoUtils objects, pass the
file path to the {class}`~geoutils.Raster`, {class}`~geoutils.Vector` or {class}`~geoutils.PointCloud` constructors.

Finally, by default, **an operation always returns an object with the same data structure**. An accessor operation will return an object type also fit for an accessor (Xarray, GeoPandas), while a GeoUtils object will return a GeoUtils object. 
Object types should not be mixed together. When ambiguity exists for return types (e.g., for point clouds), you can set which object is returned within the function, or by default in our global {ref}`config`.


(core-datasets)=
## {class}`~xarray.DataArray` or {class}`~xarray.Dataset` accessors?

Use accessors on a **{class}`~xarray.Dataset` to process several {class}`~xarray.DataArray` variables at once**.

```{tip}
:class: margin

You can pass `as_object_type="dataset"` to {func}`~geoutils.open_raster` or {func}`~geoutils.open_pointcloud` to return a {class}`~xarray.Dataset` directly (defaults to a {class}`~xarray.DataArray`).
```

```{code-cell} ipython3
:tags: [remove-cell]

# Show only the exception type and message for the expected error below
get_ipython().InteractiveTB.set_mode(mode="Minimal")
```

An example of dataset with multiple raster and point variables is:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for creating the example data"
:  code_prompt_hide: "Hide the code for creating the example data"

from affine import Affine
import numpy as np
import xarray as xr

import geoutils as gu

# Create a small raster with 10 m pixels for the examples
grid = gu.raster.from_array(
    np.arange(16, dtype=float).reshape(4, 4), Affine(10, 0, 0, 0, -10, 40), 32631
)

# We create point elevations, with separate X/Y coordinate names from the rasters
point_elevations = gu.pointcloud.from_xyz(
    x=[5.0, 15.0, 25.0], y=[25.0, 15.0, 5.0], z=[100.0, 110.0, 120.0], crs=32631
).rename({"x": "x_point", "y": "y_point"})

# We use a dataset of two raster variables on a shared grid
ds = xr.Dataset({"dem": grid, "shifted_dem": grid + 1, "height": point_elevations, "description": "Example grid"})
ds
```

The {class}`rst <geoutils.DatasetRasterAccessor>` and {class}`pc <geoutils.DatasetPointCloudAccessor>` accessors
automatically detect **compatible raster/point variables** in the {class}`~xarray.Dataset`, listed through {attr}`~geoutils.DatasetRasterAccessor.compatible_variables`, 
which corresponds to variables with the right shapes (2D or 3D for rasters, 1D for points) and spatial referencing:

```{code-cell} ipython3
list(ds.rst.compatible_variables.keys()), list(ds.pc.compatible_variables.keys())
```

**By default, operations will loop through every compatible variable**, but those can also be passed manually through the `variables`, 
which is exposed for all functions with the {class}`rst <geoutils.DatasetRasterAccessor>` Dataset accessor.

```{code-cell} ipython3
# Filtering runs through all variables by default
ds_both_filtered = ds.rst.filter("mean", window=3)

# Filtering runs only on "dem" 
ds_dem_filtered = ds.rst.filter("mean", window=3, variables=["dem"])
```

```{caution}
:class: margin

The {class}`~xarray.Dataset` accessors are recent and might evolve before the 1.0 release, but are likely to **support more diverse cases**, thus keeping backwards-compatibility. 
```

**Operations with a raster or point output** (such as {ref}`transformations` or {ref}`filters`), return a 
{class}`~xarray.Dataset` updated for each compatible variable. 

**Operations with other outputs** return a dictionary of output for each variable, except for specific cases where those outputs can 
be concatenated as input, such as for grouped statistics with {meth}`~geoutils.DatasetRasterAccessor.stats`.

As a {class}`~xarray.Dataset` can hold raster and point variables on independent dimensions, both 
{class}`rst <geoutils.DatasetRasterAccessor>` and {class}`pc <geoutils.DatasetPointCloudAccessor>` can each act on their
compatible variables.

```{code-cell} ipython3
# Filtering with "rst" runs through raster variables only
ds_rasters_filtered = ds.rst.filter("mean", window=3)

# Filtering with "pc" runs through point variables only
ds_points_filtered = ds.pc.filter("mean", radius=15)
```

However, variables that share coordinates **must** change dimensions together, otherwise a `ValueError` is raised for ambiguous behaviour.

```{code-cell} ipython3
:tags: [raises-exception]

# Cropping only "dem" changes the grid shared with "shifted_dem"
ds.rst.icrop((0, 0, 2, 2), variables=["dem"])
```

(core-point-accessors)=
## Xarray or GeoPandas for points?

Both point accessors {class}`pc <geoutils.DataArrayPointCloudAccessor>` (for Xarray) and {class}`pc <geoutils.GeoDataFramePointCloudAccessor>` (for GeoPandas) expose the same operations, 
and work on the same file formats.

Their core difference lies in the fact that **GeoPandas operations have to systematically construct [Shapely point objects](https://shapely.readthedocs.io/en/stable/reference/shapely.Point.html)**, 
even for operations in which this geometry is not utilized and could run on the original X/Y/Z coordinates instead (e.g. conversion from raster to points, 
downsampling, cropping, gridding, etc).
Our {ref}`benchmarking` shows that this can slow down operations on large point clouds.

Therefore, we advise to use the **Xarray {class}`pc <geoutils.DataArrayPointCloudAccessor>` accessor for numerical analysis** that mostly manipulates values and coordinate arrays, 
and the **GeoPandas {class}`pc <geoutils.GeoDataFramePointCloudAccessor>` for geometric operations**, such as buffers and spatial joins.

```{note}
Using {func}`~geoutils.open_pointcloud` returns a {class}`~xarray.DataArray` by default, and a {class}`~geopandas.GeoDataFrame` with `as_object_type="geodataframe"`.
```

For example, below we **time both ways** of opening a LAZ point cloud:

```{code-cell} ipython3
import geoutils as gu

# Open the example file as a DataArray with numeric X/Y coordinates and elevations
filename_pc = gu.examples.get_path("coromandel_lidar")
%timeit -n 3 -r 3 gu.open_pointcloud(filename_pc)
```

```{code-cell} ipython3
# Open the same file as a GeoDataFrame, constructing a geometry for every point
%timeit -n 3 -r 3 gu.open_pointcloud(filename_pc, as_object_type="geodataframe")
```

The Xarray option is faster!

(core-geoutils-objects)=
## GeoUtils objects

GeoUtils defines **its own geospatial objects**, created before the emergence of accessors. They associate values with geospatial 
metadata, and preserve their geospatial type through many operations. 
A {class}`~geoutils.Raster` stores a NumPy masked array, while {class}`~geoutils.Vector` and {class}`~geoutils.PointCloud` store GeoPandas
dataframes, with a {class}`~geoutils.PointCloud`'s {attr}`~PointCloudBase.data_name` selecting its main numerical values.

Their behaviour **follows similar principles to Xarray**:

- **{ref}`Deferred I/O and implicit loading <core-lazy-load>`:** When opened from files, all three objects read
  only metadata, and load data when requested. Subsetting and downsampling read only the requested data, even
  without chunks.
- **{ref}`Arithmetic interface <core-numerical-interface>`:** Rasters and point clouds expose Python operators
  and NumPy functions to perform easy arithmetic.

GeoPandas methods are available directly from {class}`~geoutils.Vector` and {class}`~geoutils.PointCloud`. Their
outputs are cast according to their type: geometric outputs remain GeoUtils objects, while non-geometric outputs keep
their original type.

In addition, those objects support file-to-file chunked execution through **{ref}`Multiprocessing <multiprocessing>`**,
while accessors use Dask instead.

See {ref}`raster-class`, {ref}`vector-class`, and {ref}`point-cloud` pages for detailed descriptions of each
object, or {ref}`feature-overview` for operations shared across data types.

(core-inheritance)=
## Inheritance

Inheritance is practical to naturally pass down parent methods and attributes to child classes.

**Within GeoUtils**, `RasterBase` defines the geospatial methods shared by {class}`~geoutils.Raster` and the Xarray
{class}`rst <geoutils.DataArrayRasterAccessor>` accessor. Both inherit from this base class, implementing
the attributes, I/O and copying needed for their NumPy or Xarray data. 
The {class}`rst <geoutils.DatasetRasterAccessor>` Dataset accessor delegates to the DataArray accessor for
each compatible variable, as described in {ref}`core-datasets`. The same logic applies to vectors and point clouds.

```{eval-rst}
.. inheritance-diagram:: geoutils.raster.raster.Raster geoutils.raster.xr_accessor.DataArrayRasterAccessor
    :top-classes: geoutils.raster.base.RasterBase
    :parts: 1
```

**This inheritance extends to other packages**, such as [xDEM](https://xdem.readthedocs.io/) for analyzing digital elevation models.

xDEM extends both interfaces: {class}`~xdem.DEM` inherits from Raster, while its `DEMAccessor` inherits from
DataArrayRasterAccessor to expose the `dem` Xarray accessor. Both also inherit from `DEMBase`, which adds
the shared elevation methods and vertical referencing. The same logic applies to elevation point clouds.

```{graphviz}
digraph dem_inheritance {
    graph [rankdir=LR, bgcolor="white"];
    node [shape=box, fontname="Arial", fontsize=10];
    edge [color="dodgerblue1", arrowsize=0.7];

    RasterBase [label="RasterBase"];
    Raster [label="Raster"];
    RasterAccessor [label="DataArrayRasterAccessor"];
    DEMBase [label="DEMBase (xDEM)"];
    DEM [label="DEM (xDEM)"];
    DEMAccessor [label="DEMAccessor (xDEM)"];

    RasterBase -> Raster;
    RasterBase -> RasterAccessor;
    RasterBase -> DEMBase;
    Raster -> DEM;
    RasterAccessor -> DEMAccessor;
    DEMBase -> DEM;
    DEMBase -> DEMAccessor;
}
```

```{note}
The `dem` accessor and {class}`~xdem.DEM` class notably re-implements all methods of [gdalDEM](https://gdal.org/programs/gdaldem.html) to derive topographic attributes
(hillshade, slope, aspect, etc), and much more.

If you are DEM-enthusiastic, **[check-out our sister package xDEM](https://xdem.readthedocs.io/) on elevation data analysis.**
```

Many types of geospatial data can be viewed as a subclass of rasters or point clouds, which have more attributes and require their own methods:
**spectral images**, **velocity fields**, **phase difference maps**, etc...

If you are interested to build your own subclass of a GeoUtils object or accessor: you can take example from xDEM, 
and don't hesitate to reach out on our GitHub!
