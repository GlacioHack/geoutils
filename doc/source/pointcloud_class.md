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
(point-cloud)=

# The georeferenced point cloud ({class}`~geoutils.PointCloud`)

A point cloud represents 2D point geometries of georeferenced coordinates associated with a main 1D data array, and optionally auxiliary data.

Although a subtype of {class}`~geoutils.Vector`, point clouds have a very different nature than other vectors and are
ubiquitous in geospatial analysis, requiring their own object type.
For **numerical operations**, a point cloud manipulation is facilitated by its own arithmetic to manipulate its main data array, as well as a
specific interface with rasters (interpolation or reduction to same coordinates, or point gridding), or other point
clouds (pairwise distance matching).
For **geometric operations**, point clouds interface in specific ways with other vector-types (zonal statistics, geometric masking).

GeoUtils aims to support these features, as well the reading and writing of point clouds both from vector-type files (e.g., ESRI shapefile, geopackage,
geoparquet) usually used for **sparse point clouds**, and from point-cloud-type files (e.g., LAS, LAZ, COPC) usually
used for **dense point clouds**.

```{warning}
Support for LAS files is still preliminary and loads all data in memory for most operations. We are working on adding operations with chunked reading.
```

Below, a summary of the {class}`~geoutils.PointCloud` object and its methods.

(pc-obj-def)=

## Object definition and attributes

A {class}`~geoutils.PointCloud` is a {class}`~geoutils.Vector` is a vector of 2D point geometries associated to
numeric values from a main {attr}`~geoutils.PointCloud.data` column, and can also contain auxiliary data columns.

It inherits the main {class}`~geoutils.Vector` attribute {attr}`~geoutils.Vector.ds` containing the geodataframe, and adds **another
main attribute** {attr}`~geoutils.PointCloud.data_column` that identifies the name of the main data associated to the
point geometries.

Additionally, new attributes such as {attr}`~geoutils.PointCloud.point_count` and new methods specific to point clouds are detailed further below.

Generic vector attributes and methods are inherited through the {class}`~geoutils.Vector` object, such as
{attr}`~geoutils.Vector.bounds`, {attr}`~geoutils.Vector.crs`, {func}`~xdem.Vector.reproject` and {func}`~xdem.Vector.crop`.

```{tip}
The complete list of {class}`~geoutils.Vector` attributes and methods can be found in [the Vector section of the API](https://geoutils.readthedocs.io/en/stable/api.html#vector).
```

## Open and save

A {class}`~geoutils.PointCloud` is opened by instantiating the class with a {class}`str`, a {class}`pathlib.Path`, a {class}`geopandas.GeoDataFrame`,
a {class}`geopandas.GeoSeries` or a {class}`shapely.Geometry`.

```{code-cell} ipython3
:tags: [remove-cell]

# To get a good resolution for displayed figures
from matplotlib import pyplot
pyplot.rcParams['figure.dpi'] = 400
pyplot.rcParams['savefig.dpi'] = 400
```

```{code-cell} ipython3
import geoutils
import numpy as np

# Instantiate a point cloud from a filename on disk
filename_dem = geoutils.examples.get_path("coromandel_lidar")
pc = geoutils.PointCloud(filename_dem, data_column="Z")
pc
```

## Create from arrays or tuples

A {class}`~geoutils.PointCloud` is created from three 1D arrays, from a Nx3 or 3xN array, or from an iterable of 3-tuples by calling the class
methods {func}`~geoutils.PointCloud.from_xyz`, {func}`~geoutils.PointCloud.from_array` or {func}`~geoutils.PointCloud.from_tuples` respectively.

```{code-cell} ipython3
# From three 1D arrays
pc1 = geoutils.PointCloud.from_xyz(x=np.array([1, 4]), y=np.array([2, 5]), z=np.array([3, 6]), crs=4326)
# From a Nx3 array
pc2 = geoutils.PointCloud.from_array(np.array([[1, 2, 3], [4, 5, 6]]), crs=4326)
# From a iterable of 3-tuples
pc3 = geoutils.PointCloud.from_tuples([(1, 2, 3), (4, 5, 6)], crs=4326)
```

## Gridding to a raster

Gridding a {class}`~geoutils.PointCloud` into a specific grid can be done through the {func}`~geoutils.PointCloud.grid` function, that applies a
gridding scheme according to a given methods (inverse-distance weighting, delauney triangle interpolation).

```{code-cell} ipython3
# Grid the point cloud on a 100x100 grid on its extent
coords = (np.linspace(pc.bounds.left, pc.bounds.right, 100), np.linspace(pc.bounds.bottom, pc.bounds.top, 100))
rst = pc.grid(grid_coords=coords)
```

## Arithmetic


A {class}`~geoutils.PointCloud` can be applied any pythonic arithmetic operation ({func}`+<operator.add>`, {func}`-<operator.sub>`, {func}`/<operator.truediv>`, {func}`//<operator.floordiv>`, {func}`*<operator.mul>`,
{func}`**<operator.pow>`, {func}`%<operator.mod>`) with another {class}`~geoutils.PointCloud`, {class}`~numpy.ndarray` or number. It will output one or two
{class}`PointClouds<geoutils.PointCloud>`. NumPy coercion rules apply for {class}`dtype<numpy.dtype>`.
The operation is applied to the {attr}`~geoutils.PointCloud.data_column` of the point cloud.

```{code-cell} ipython3
# Add 1 and divide point cloud by 2
(pc1 + 1)/2
```

A {class}`~geoutils.PointCloud` can also be applied any pythonic logical comparison operation ({func}`==<operator.eq>`, {func}` != <operator.ne>`,
{func}`>=<operator.ge>`, {func}`><operator.gt>`, {func}`<=<operator.le>`, {func}`<<operator.lt>`) with another {class}`~geoutils.PointCloud`,
{class}`~numpy.ndarray` or number. It will cast to a boolean {class}`~geoutils.PointCloud`.

```{code-cell} ipython3
# What are point cloud pixels are larger than 20?
pc1 > 20
```

See {ref}`core-py-ops` for more details.

## Array interface

A {class}`~geoutils.PointCloud` can be applied any NumPy universal functions and most mathematical, logical or masked-array functions with another
{class}`~geoutils.PointCloud`, {class}`~numpy.ndarray` or number.
The operation is applied to the {attr}`~geoutils.PointCloud.data_column` of the point cloud.

```{code-cell} ipython3
# Compute the element-wise square-root
np.sqrt(pc1)
```

Logical comparison functions will cast to a boolean {class}`~geoutils.PointCloud`.

```{code-cell} ipython3
# Is the pointcloud close to another one within tolerance?

np.isclose(pc1, pc1+0.05, atol=0.1)
```

See {ref}`core-array-funcs` for more details.

## Statistics

Statistics of a point cloud, optionally subsetting to an inlier mask, can be computed using {func}`~geoutils.PointCloud.get_stats`.

```{code-cell} ipython3
# Get mean, max and STD of the point cloud
pc.get_stats(["mean", "max", "std"])
```

A point cloud can also be quickly subsampled using {func}`~geoutils.PointCloud.subsample`, which considers only valid values, and returns either a point
cloud or an array:

```{code-cell} ipython3
# Get 500 random points in the point cloud
pc_sub = pc.subsample(500)
```

See {ref}`stats` for more details.
