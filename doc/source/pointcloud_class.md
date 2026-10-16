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

# The georeferenced point cloud

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
numeric values from a named {attr}`~geoutils.PointCloud.data` attribute, and can also contain auxiliary attributes.

It inherits the main {class}`~geoutils.Vector` attribute {attr}`~geoutils.Vector.gdf` containing the geodataframe, and adds **another
main attribute** {attr}`~geoutils.PointCloud.data_name` that identifies the name of the main data associated to the
point geometries.

Additionally, new attributes such as {attr}`~geoutils.PointCloud.point_count` and new methods specific to point clouds are detailed further below.

Generic vector attributes and methods are inherited through the {class}`~geoutils.Vector` object, such as
{attr}`~geoutils.Vector.bbox`, {attr}`~geoutils.Vector.crs`, {func}`~geoutils.Vector.reproject` and
{func}`~geoutils.Vector.crop`.

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
pc = geoutils.PointCloud(filename_dem, data_name="Z")
pc
```

### Remote files

Pass an HTTP URL or an S3 URI to {func}`~geoutils.open_pointcloud` or {class}`~geoutils.PointCloud` to read remote
LAS, LAZ, COPC and GeoParquet files. Install `geoutils[cloud]` for the optional HTTP and S3 clients. GeoParquet also
requires `geoutils[parquet]`, and LAS/LAZ requires LasPy with a decompression backend for compressed files.

```python
points = geoutils.open_pointcloud("s3://my-bucket/points.copc.laz", downsample=10, chunks=100_000)
```

Native readers use [fsspec's buffered range reads](https://filesystem-spec.readthedocs.io/en/latest/features.html#file-buffering-and-random-access)
and [backend configuration](https://filesystem-spec.readthedocs.io/en/latest/config.html) for credentials and endpoint
options. HTTP query strings are passed through unchanged, including signed URLs. With `chunks`, workers open their
own streams when computing the result. COPC depth selection and automatic downsampling work through the same URLs;
S3 GeoParquet datasets may also contain multiple files under one prefix.

### COPC levels of detail

[COPC files](https://copc.io/) store points at several octree depths. Smaller depths contain coarser samples. Pass `level` to
{func}`~geoutils.open_pointcloud` to select depths using
[LasPy's conventions](https://laspy.readthedocs.io/en/latest/api/laspy.copc.html): an integer selects one depth,
and a range selects several. For example:

```python
# Read the root sample
points = geoutils.open_pointcloud("points.copc.laz", level=0)

# Read all points stored at depths 0 through 2
points = geoutils.open_pointcloud("points.copc.laz", level=range(3), chunks=100_000)
```

With `downsample`, GeoUtils reads hierarchy counts to choose the coarsest cumulative depths containing at least
`ceil(N / downsample)` points, where `N` is the original file's point count. It then takes a deterministic random
sample of those points to reach that count.
This mirrors automatic raster overview selection: stored reduced data supplies the requested output size.

Explicit `level` and `downsample > 1` cannot be combined. Depth selection works for DataArray, Dataset and
GeoDataFrame output. With `chunks`, opening reads metadata only; selected point records load when the result is
computed. Ordinary LAS and LAZ files use the existing random sampling and do not accept `level`.

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

Gridding a {class}`~geoutils.PointCloud` into a specific grid can be done through the {func}`~geoutils.PointCloud.grid` function, using nearest,
linear or cubic interpolation, or circular inverse-distance weighting and statistics. Circular statistics include mean,
minimum, maximum, range, count, population standard deviation, average point-to-cell distance and average spacing
between points. Invalid point coordinates or values are ignored, and cells without the required number of finite
neighbors remain nodata.

```{code-cell} ipython3
# Grid the point cloud on a 100x100 grid on its extent
coords = (np.linspace(pc.bbox.left, pc.bbox.right, 100), np.linspace(pc.bbox.bottom, pc.bbox.top, 100))
rst = pc.grid(grid_coords=coords, resampling="idw", dist_nodata_pixel=3)
```

Use `min_points` to require several finite neighbors in the circular support, for example
`pc.grid(..., resampling="mean", min_points=3)`.
The default `engine="scipy"` supports every method. `engine="numba"` provides an optional compiled implementation for
nearest and circular methods except `average_distance_pts`, which can be faster depending on the number of points.

## Arithmetic


A {class}`~geoutils.PointCloud` can be applied any pythonic arithmetic operation ({func}`+<operator.add>`, {func}`-<operator.sub>`, {func}`/<operator.truediv>`, {func}`//<operator.floordiv>`, {func}`*<operator.mul>`,
{func}`**<operator.pow>`, {func}`%<operator.mod>`) with another {class}`~geoutils.PointCloud`, {class}`~numpy.ndarray` or number. It will output one or two
{class}`PointClouds<geoutils.PointCloud>`. NumPy coercion rules apply for {class}`dtype<numpy.dtype>`.
The operation is applied to the {attr}`~geoutils.PointCloud.data_name` of the point cloud.

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
The operation is applied to the {attr}`~geoutils.PointCloud.data_name` of the point cloud.

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

Statistics of a point cloud can be computed using {func}`~geoutils.PointCloud.stats`.

```{code-cell} ipython3
# Get mean, max and STD of the point cloud
pc.stats(["mean", "max", "std"])
```

A point cloud can also be quickly subsampled using {func}`~geoutils.PointCloud.subsample`, which considers only valid
values and returns the complete selected point rows by default:

```{code-cell} ipython3
# Get 500 random points
pc_sub = pc.subsample(500)

# Return only their values as an array
values_sub = pc.subsample(500, as_array=True)
```

See {ref}`stats` for more details.
