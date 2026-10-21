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

In GeoUtils, georeferenced point clouds are exposed through the following interfaces:

- The Xarray {class}`pc <geoutils.DataArrayPointCloudAccessor>` accessor for a {class}`xarray.DataArray`,
- The GeoPandas {class}`pc <geoutils.GeoDataFramePointCloudAccessor>` accessor for a {class}`geopandas.GeoDataFrame`,
- The {class}`~geoutils.PointCloud`.

For new workflows, we **recommend using the accessors**, and using either only accessors or only GeoUtils objects.

A point cloud represents 2D point geometries of georeferenced coordinates associated with a main 1D data array, and optionally auxiliary data.

```{important}
The {class}`pc <geoutils.DatasetPointCloudAccessor>` accessor on a {class}`~xarray.Dataset` supports **processing several point clouds at once**. See {ref}`core-datasets`.
```

Except for the differences outlined in {ref}`pc-accessor-comparison`, the accessors and the {class}`~geoutils.PointCloud`
object provide the **same geospatial operations**, described in the feature pages linked in {ref}`feature-overview`
and listed in {ref}`pointcloud-api`.

Below, we describe how to **open, create and save point clouds with the accessors**, followed by the **attributes and
behaviour of the {class}`~geoutils.PointCloud` object**.

(pc-accessor-comparison)=
## Accessors {class}`pc <geoutils.DataArrayPointCloudAccessor>` versus {class}`~geoutils.PointCloud`

The main differences are the following:

1. A {class}`~geoutils.PointCloud` applies **arithmetic to its main data column**, while the accessors require selecting the column or variable explicitly.

2. A {class}`~geoutils.PointCloud` supports **Multiprocessing**, while the accessors support **Dask**, allowing lazy graph building.

For numerical analysis, we recommend the **Xarray accessor**, and for geometric operations, the **GeoPandas accessor**. See {ref}`core-point-accessors` for details.

## Why a separate point object?

Although a subtype of {class}`~geoutils.Vector`, point clouds have a very different nature than other vectors and are
ubiquitous in geospatial analysis, requiring their own object type.
For **numerical operations**, a point cloud manipulation is facilitated by its own arithmetic to manipulate its main data array, as well as a
specific interface with rasters (interpolation or reduction to same coordinates, or point gridding), or other point
clouds (pairwise distance matching).
For **geometric operations**, point clouds interface in specific ways with other vector-types (zonal statistics, geometric masking).

GeoUtils supports these features, as well the reading and writing of point clouds both from vector-type files (e.g., ESRI shapefile, GeoPackage,
GeoParquet) usually used for **sparse point clouds**, and from point-cloud-type files (e.g., LAS, LAZ, COPC) usually
used for **dense point clouds** like LiDAR.

## Point cloud accessors

### Opening files

A point cloud is opened with {func}`~geoutils.open_pointcloud`, which returns a {class}`~xarray.DataArray` by default.
Its values contain the main measurement, while auxiliary attributes are stored as coordinates along the same point dimension.

```{code-cell} ipython3
import geoutils

filename_dem = geoutils.examples.get_path("coromandel_lidar")

# Open elevation and classification as lazy numeric arrays from the LAS file
points = geoutils.open_pointcloud(filename_dem, data_name="Z", columns=["classification"], chunks=5000)
points
```

Passing `as_object_type="dataset"` stores several measurements as separate variables, while `as_object_type="geodataframe"`
returns a {class}`~geopandas.GeoDataFrame` for working with point geometries. For LAS and GeoParquet files, `columns`
selects the main attributes (`"main"`), all attributes (`"all"`) or a list of names. The CRS, X/Y coordinates and main
measurement are always read together.

By default, the Xarray data is read only when needed (`load=False`), without requiring Dask. We can select points with
`isel(point=...)` before loading their values. This reads only the selected LAS/LAZ records, intersecting COPC nodes or
intersecting GeoParquet row groups. Passing `chunks=` enables Dask execution and sets the number of points per chunk,
while `load=True` reads the data into memory when opening the file.

For a {class}`~geopandas.GeoDataFrame`, `load=False` uses Dask-GeoPandas, and `chunks=` sets the number of points per
partition. Passing `parse_sensor_metadata=True` adds sensor information parsed from the filename.

### Constructing from arrays

A point cloud is created from X/Y coordinate and value arrays with {func}`~geoutils.pointcloud.from_xyz`, which also
accepts auxiliary attributes:

```python
points = geoutils.pointcloud.from_xyz([0, 1], [2, 3], [4, 5], crs=32631, data_name="height")
```

As with {func}`~geoutils.open_pointcloud`, passing `as_object_type="dataset"` or `"geodataframe"` returns the corresponding
object type. Dask arrays remain lazy during construction.

### Writing points

A point cloud is saved to file with {meth}`~geoutils.DataArrayPointCloudAccessor.to_file`,
{meth}`~geoutils.DataArrayPointCloudAccessor.to_las` or {meth}`~geoutils.DataArrayPointCloudAccessor.to_parquet`.
For GeoParquet, a lazy array can be written in row groups of a chosen size or as a directory of partition files:

```python
# Write lazy point arrays in groups of 5000 rows
points.pc.to_parquet("elevations.parquet", chunks=5000)

# Reopen the saved active measurement and auxiliary attributes
reopened = geoutils.open_pointcloud("elevations.parquet", columns="all", chunks=5000)
```

Writing runs the tasks needed to save the data, including for lazy arrays. The GeoParquet file stores the point
coordinates, CRS and main measurement. See {ref}`scalability-formats` for choosing formats and partition sizes,
and {ref}`core-composition` for converting between array and geometry interfaces.

## The {class}`~geoutils.PointCloud` object

(pc-obj-def)=
### Object definition and attributes

A {class}`~geoutils.PointCloud` is a vector of 2D point geometries associated with
numeric values from a named {attr}`~PointCloudBase.data` attribute, and can also contain auxiliary attributes.

It inherits the main {class}`~geoutils.Vector` attribute {attr}`~geoutils.Vector.gdf` containing the geodataframe, and adds **another
main attribute** {attr}`~PointCloudBase.data_name` that identifies the name of the main data associated to the
point geometries.

Additionally, new attributes such as {attr}`~geoutils.pointcloud.base.GeometryPointCloudBase.point_count` and new methods specific to point clouds are detailed further below.

Generic vector attributes and methods are inherited through the {class}`~geoutils.Vector` object, such as
{attr}`~VectorBase.bbox`, {attr}`~VectorBase.crs`, {meth}`~VectorBase.reproject` and
{meth}`~VectorBase.crop`.

```{tip}
The complete list of {class}`~geoutils.Vector` attributes and methods can be found in {ref}`the Vector section of the API <vector-api>`.
```

### Open and save

A {class}`~geoutils.PointCloud` is opened by instantiating the class with a {class}`str`, a {class}`pathlib.Path`, a {class}`geopandas.GeoDataFrame`,
a {class}`geopandas.GeoSeries` or a {external+shapely:doc}`shapely.Geometry <geometry>`.

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

See {ref}`core-downsampling` for details on reducing point cloud density when opening or plotting,
including how GeoUtils selects deterministic point samples.

### Create from arrays or tuples

A {class}`~geoutils.PointCloud` is created from three 1D arrays, from a Nx3 or 3xN array, or from an iterable of 3-tuples by calling the class
methods {meth}`~PointCloudBase.from_xyz`, {meth}`~PointCloudBase.from_array` or {meth}`~PointCloudBase.from_tuples` respectively.

```{code-cell} ipython3
# From three 1D arrays
pc1 = geoutils.PointCloud.from_xyz(x=np.array([1, 4]), y=np.array([2, 5]), z=np.array([3, 6]), crs=4326)
# From a Nx3 array
pc2 = geoutils.PointCloud.from_array(np.array([[1, 2, 3], [4, 5, 6]]), crs=4326)
# From a iterable of 3-tuples
pc3 = geoutils.PointCloud.from_tuples([(1, 2, 3), (4, 5, 6)], crs=4326)
```

### Arithmetic


A {class}`~geoutils.PointCloud` can be applied any pythonic arithmetic operation ({func}`+<operator.add>`, {func}`-<operator.sub>`, {func}`/<operator.truediv>`, {func}`//<operator.floordiv>`, {func}`*<operator.mul>`,
{func}`**<operator.pow>`, {func}`%<operator.mod>`) with another {class}`~geoutils.PointCloud`, {class}`~numpy.ndarray` or number. It will output one or two
{class}`PointClouds<geoutils.PointCloud>`. NumPy coercion rules apply for {class}`dtype<numpy.dtype>`.
The operation is applied to the {attr}`~PointCloudBase.data_name` of the point cloud.

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

### Array interface

A {class}`~geoutils.PointCloud` can be applied any NumPy universal functions and most mathematical, logical or masked-array functions with another
{class}`~geoutils.PointCloud`, {class}`~numpy.ndarray` or number.
The operation is applied to the {attr}`~PointCloudBase.data_name` of the point cloud.

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

## Related features

For more details on features applicable to point clouds, refer to the following pages!

| Topic | Documentation |
|---|---|
| CRS, bounds, and footprints | {ref}`referencing` |
| Reprojection, cropping, clipping, interpolation, reduction, and gridding | {ref}`transformations` |
| Subsampling, co-sampling, and pair sampling | {ref}`sampling` |
| Global, grouped, and spatial statistics | {ref}`stats` |
| Neighborhood filtering and missing values | {ref}`filters-points` |
| Error models, uncertainty propagation and simulation | {ref}`uncertainty` |
| Native point formats and partitioned writing | {ref}`scalability-formats` |
| Downsampling for opening and plotting | {ref}`core-downsampling` |
| Lazy and out-of-memory execution | {ref}`scalability-index` |
| Complete method and attribute listing | {ref}`pointcloud-api` |
