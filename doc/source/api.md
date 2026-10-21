(api)=
# API reference

This page provides a summary of GeoUtils’ API.
For more details and examples, refer to the relevant chapters in the main part of the documentation.

```{eval-rst}
.. currentmodule:: geoutils
```


<!--
Hidden toctree so object and accessor pages are generated and discoverable via cross-references, without duplicating
the main API method summaries. This needs to be listed before any other API.
-->
```{toctree}
:maxdepth: 1
:hidden:

DataArray.rst accessor <api_rst>
Dataset.rst accessor <api_rst_dataset>
Raster <api_raster>
GeoDataFrame.vct accessor <api_vct>
Vector <api_vector>
DataArray.pc accessor <api_pc_dataarray>
Dataset.pc accessor <api_pc_dataset>
GeoDataFrame.pc accessor <api_pc>
PointCloud <api_pointcloud>
Inherited Vector methods <api_vector_inherited>
```

(raster-api)=
## Raster API

GeoUtils exposes the raster API through the following interfaces:

- A {class}`DataArray.rst <geoutils.DataArrayRasterAccessor>` accessor extending {class}`xarray.DataArray` objects as rasters.
- A {class}`Dataset.rst <geoutils.DatasetRasterAccessor>` accessor applying raster methods to compatible variables in {class}`xarray.Dataset` objects.
- {class}`~geoutils.Raster`, an interface operating directly on a GeoUtils object.


The DataArray accessor and Raster expose the **same methods and attributes**. The Dataset accessor applies operations
to all compatible raster variables by default, or to those selected.

Only **file opening** and **scalable execution** differ substantially between the accessors and Raster:

- **File opening:** {meth}`~geoutils.open_raster` is used for an {class}`xarray.DataArray` object, while {class}`~geoutils.Raster` objects are opened by instantiating the class,
- **Scalable execution:** the {class}`DataArray.rst <geoutils.DataArrayRasterAccessor>` accessor supports **Dask**, while the {class}`~geoutils.Raster` supports **Multiprocessing** instead.

### Opening a raster file

Use {meth}`~geoutils.open_raster` for an {class}`xarray.DataArray`, or instantiate for a {class}`~geoutils.Raster`.

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    open_raster
    Raster.__init__
```

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: raster_method.rst

    ~raster.base.RasterBase.info
```

### Create raster from an array

Use {func}`~geoutils.raster.from_array` to create a DataArray, or pass `as_object_type="dataset"` to create a Dataset.

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    ~raster.from_array
```

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: raster_method.rst

    ~raster.base.RasterBase.from_array
```

(api-raster-attrs)=

### Main attributes

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: raster_method.rst

    ~raster.base.RasterBase.data
    ~raster.base.RasterBase.crs
    ~raster.base.RasterBase.transform
    ~raster.base.RasterBase.nodata
    ~raster.base.RasterBase.area_or_point
```

### Derived attributes

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: raster_method.rst

    ~raster.base.RasterBase.shape
    ~raster.base.RasterBase.height
    ~raster.base.RasterBase.width
    ~raster.base.RasterBase.count
    ~raster.base.RasterBase.bands
    ~raster.base.RasterBase.res
    ~raster.base.RasterBase.bbox
    ~raster.base.RasterBase.footprint
    ~raster.base.RasterBase.dtype
```

### Other attributes

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: raster_method.rst

    ~raster.base.RasterBase.is_mask
    ~raster.base.RasterBase.is_loaded
    ~raster.base.RasterBase.name
    ~raster.base.RasterBase.driver
    ~raster.base.RasterBase.tags
```

(api-geo-handle)=

### Geospatial operations

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: raster_method.rst

    ~raster.base.RasterBase.crop
    ~raster.base.RasterBase.icrop
    ~raster.base.RasterBase.clip
    ~raster.base.RasterBase.reproject
    ~raster.base.RasterBase.polygonize
    ~raster.base.RasterBase.interp_at_points
    ~raster.base.RasterBase.resample_at_points
    ~raster.base.RasterBase.reduce_at_points
    ~raster.base.RasterBase.krige
    ~raster.base.RasterBase.random_field
```

### Proximity

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: raster_method.rst

    ~raster.base.RasterBase.proximity
```

### Filters

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: raster_method.rst

    ~raster.base.RasterBase.filter
    ~raster.base.RasterBase.fill_nodata
    ~raster.base.RasterBase.sieve
```

### Plotting

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: raster_method.rst

    ~raster.base.RasterBase.plot
```

(api-raster-statistics)=
### Statistics

See {ref}`stats` for estimators, grouping by intervals or categories, and variograms.

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: raster_method.rst

    ~raster.base.RasterBase.stats
    ~raster.base.RasterBase.variogram
    ~raster.base.RasterBase.estimate_error_structure
```

(api-raster-sampling)=
### Sampling

See {ref}`sampling` for selecting valid observations, common locations and spatial pairs.

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: raster_method.rst

    ~raster.base.RasterBase.subsample
    ~raster.base.RasterBase.cosample
    ~raster.base.RasterBase.pairsample
```

### Data manipulation

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: raster_method.rst

    ~raster.base.RasterBase.copy
    ~raster.base.RasterBase.to_nanarray
```

The following methods apply only to the {class}`~geoutils.Raster` object. Use native Xarray methods such as
{meth}`~xarray.DataArray.astype` and {meth}`~xarray.DataArray.where` with the accessor interface.

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Raster.astype
    Raster.set_mask
    Raster.get_mask
```

### Loading, writing and converting

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: raster_method.rst

    ~raster.base.RasterBase.load
    ~raster.base.RasterBase.to_pointcloud
    ~raster.base.RasterBase.from_pointcloud_regular
```

Writing to file from accessors or Raster object:

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DataArrayRasterAccessor.to_file
    DatasetRasterAccessor.to_file
    DataArrayRasterAccessor.to_geoutils
    Raster.to_file
    Raster.to_rio_dataset
    Raster.to_xarray
```

### Georeferencing utilities

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: raster_method.rst

    ~raster.base.RasterBase.edit
    ~raster.base.RasterBase.set_crs
    ~raster.base.RasterBase.set_transform
    ~raster.base.RasterBase.set_nodata
    ~raster.base.RasterBase.set_area_or_point
    ~raster.base.RasterBase.xy2ij
    ~raster.base.RasterBase.ij2xy
    ~raster.base.RasterBase.coords
    ~raster.base.RasterBase.translate
    ~raster.base.RasterBase.outside_image
```

### Projection utilities

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: raster_method.rst

    ~raster.base.RasterBase.get_metric_crs
    ~raster.base.RasterBase.get_bbox_projected
    ~raster.base.RasterBase.get_footprint_projected
    ~raster.base.RasterBase.intersection
```

### Testing utilities

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: raster_method.rst

    ~raster.base.RasterBase.raster_equal
    ~raster.base.RasterBase.georeferenced_grid_equal
```

### Multiple rasters

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    raster.load_multiple_rasters
    raster.stack_rasters
    raster.merge_rasters
```

(vector-api)=
## Vector API

GeoUtils exposes the vector API through two mirrored interfaces:

- A {class}`vct <geoutils.GeoDataFrameVectorAccessor>` accessor extending {class}`geopandas.GeoDataFrame` objects as vectors.
- {class}`~geoutils.Vector`, an interface operating directly on a GeoUtils object.

The GeoDataFrame accessor and Vector expose the **same methods and attributes**. Native GeoPandas
methods are available directly on the GeoDataFrame.

Only **file opening** and **scalable execution** differ substantially between the accessor and Vector:

- **File opening:** {meth}`~geoutils.open_vector` is used for a {class}`~geopandas.GeoDataFrame` object, while {class}`~geoutils.Vector` objects are opened by instantiating the class,
- **Scalable execution:** the {class}`vct <geoutils.GeoDataFrameVectorAccessor>` accessor supports **Dask**, while the {class}`~geoutils.Vector` supports **Multiprocessing** instead.

### Opening a file

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    open_vector
    Vector.__init__
```

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: vector_method.rst

    ~vector.base.VectorBase.info
```

### Main attributes

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Vector.gdf

.. autosummary::
    :toctree: gen_modules/
    :template: vector_method.rst

    ~vector.base.VectorBase.crs
    ~vector.base.VectorBase.bbox
    ~vector.base.VectorBase.footprint
    ~vector.base.VectorBase.name
    ~vector.base.VectorBase.is_loaded
```

```{caution}
The {attr}`~VectorBase.bbox` attribute of a {class}`~geoutils.Vector` corresponds to the {attr}`~geopandas.GeoDataFrame.total_bounds` attribute of a
{class}`~geopandas.GeoDataFrame`, while {attr}`geopandas.GeoDataFrame.bounds` returns a {class}`~pandas.Series` of per-feature bounds of the vector.
```

### Writing to file

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    GeoDataFrameVectorAccessor.to_file
    Vector.to_file
```

### Geospatial handling methods

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: vector_method.rst

    ~vector.base.VectorBase.crop
    ~vector.base.VectorBase.clip
    ~vector.base.VectorBase.reproject
    ~vector.base.VectorBase.translate
    ~vector.base.VectorBase.rasterize
```

### Plotting

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: vector_method.rst

    ~vector.base.VectorBase.plot
```

### Create geometry mask

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: vector_method.rst

    ~vector.base.VectorBase.create_mask
```

### Proximity

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: vector_method.rst

    ~vector.base.VectorBase.proximity
```

### Geometry manipulation

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: vector_method.rst

    ~vector.base.VectorBase.buffer_metric
    ~vector.base.VectorBase.buffer_without_overlap
```

### Data manipulation

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: vector_method.rst

    ~vector.base.VectorBase.copy
```

### Projection tools

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: vector_method.rst

    ~vector.base.VectorBase.get_metric_crs
    ~vector.base.VectorBase.from_bounds_projected
    ~vector.base.VectorBase.get_bbox_projected
    ~vector.base.VectorBase.get_footprint_projected
```

### Indexing

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Vector.__getitem__
```

### From Shapely and GeoPandas

Native operations remain available directly on the GeoDataFrame. The {ref}`vector-from-geopandas` appendix lists the
additional methods exposed by the {class}`~geoutils.Vector` wrapper and explains how their outputs are converted.

(pointcloud-api)=
## Point cloud API

GeoUtils exposes the point cloud API through the following interfaces:

- A {class}`DataArray.pc <geoutils.DataArrayPointCloudAccessor>` accessor extending {class}`xarray.DataArray` objects as point clouds.
- A {class}`Dataset.pc <geoutils.DatasetPointCloudAccessor>` accessor applying point cloud methods to compatible variables in {class}`xarray.Dataset` objects.
- A {class}`pc <geoutils.GeoDataFramePointCloudAccessor>` accessor extending {class}`geopandas.GeoDataFrame` objects as point clouds.
- {class}`~geoutils.PointCloud`, an interface operating directly on a GeoUtils object.

The DataArray and GeoDataFrame accessors and PointCloud expose the **same GeoUtils-specific methods and attributes**.
The Dataset accessor applies operations to all compatible point variables by default, or to those selected.
Native Xarray and GeoPandas methods remain available directly on their objects.

Only **file opening** and **scalable execution** differ substantially between the accessors and PointCloud:

- **File opening:** {meth}`~geoutils.open_pointcloud` is used for Xarray or GeoPandas objects, while {class}`~geoutils.PointCloud` objects are opened by instantiating the class,
- **Scalable execution:** the accessors support **Dask**, while the {class}`~geoutils.PointCloud` supports **Multiprocessing** instead.


### Opening a file

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    open_pointcloud
    PointCloud.__init__
```

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: pointcloud_method.rst

    ~pointcloud.base.GeometryPointCloudBase.info
```

### Main attributes

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    PointCloud.gdf

.. autosummary::
    :toctree: gen_modules/
    :template: pointcloud_method.rst

    ~pointcloud.base.PointCloudBase.data_name
    ~pointcloud.base.PointCloudBase.data
    ~pointcloud.base.PointCloudBase.crs
    ~pointcloud.base.GeometryPointCloudBase.bbox
    ~pointcloud.base.GeometryPointCloudBase.footprint
```

### Other attributes

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: pointcloud_method.rst

    ~pointcloud.base.GeometryPointCloudBase.point_count
    ~pointcloud.base.PointCloudBase.is_loaded
```


### Create and convert from data

Use {func}`~geoutils.pointcloud.from_xyz` to create a DataArray from coordinate and value arrays. Pass
`as_object_type="dataset"` or `as_object_type="geodataframe"` to choose another representation.

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    ~pointcloud.from_xyz
```

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: pointcloud_method.rst

    ~pointcloud.base.PointCloudBase.from_xyz
    ~pointcloud.base.PointCloudBase.from_array
    ~pointcloud.base.PointCloudBase.from_tuples
    ~pointcloud.base.PointCloudBase.to_xyz
    ~pointcloud.base.GeometryPointCloudBase.to_array
    ~pointcloud.base.GeometryPointCloudBase.to_tuples

```

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    PointCloud.to_xarray
```

### Writing to file

Accessor methods write the native Xarray or GeoPandas representation:

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    DataArrayPointCloudAccessor.to_file
    DataArrayPointCloudAccessor.to_las
    DataArrayPointCloudAccessor.to_parquet
    DatasetPointCloudAccessor.to_file
    DatasetPointCloudAccessor.to_las
    DatasetPointCloudAccessor.to_parquet
    GeoDataFramePointCloudAccessor.to_file
```

GeoUtils object methods:

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    PointCloud.to_file
    PointCloud.to_las
    PointCloud.to_parquet
```

### Geospatial

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: pointcloud_method.rst

    ~pointcloud.base.GeometryPointCloudBase.crop
    ~pointcloud.base.GeometryPointCloudBase.clip
    ~pointcloud.base.GeometryPointCloudBase.reproject
    ~pointcloud.base.GeometryPointCloudBase.translate
```

### Filtering

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: pointcloud_method.rst

    ~pointcloud.base.PointCloudBase.filter
```

### Interface with raster

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: pointcloud_method.rst

    ~pointcloud.base.PointCloudBase.grid
    ~pointcloud.base.PointCloudBase.krige
    ~pointcloud.base.PointCloudBase.random_field
```

### Plotting

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: pointcloud_method.rst

    ~pointcloud.base.GeometryPointCloudBase.plot
```

### Data manipulation

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: pointcloud_method.rst

    ~pointcloud.base.PointCloudBase.copy
```

(api-point-statistics)=
### Statistics

See {ref}`stats` for the same statistical workflows on point cloud values.

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: pointcloud_method.rst

    ~pointcloud.base.PointCloudBase.stats
    ~pointcloud.base.PointCloudBase.variogram
    ~pointcloud.base.PointCloudBase.estimate_error_structure
```

(api-point-sampling)=
### Sampling

See {ref}`sampling` for sampling values and locations from point clouds.

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: pointcloud_method.rst

    ~pointcloud.base.PointCloudBase.subsample
    ~pointcloud.base.PointCloudBase.cosample
    ~pointcloud.base.PointCloudBase.pairsample
```

### Testing methods

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: pointcloud_method.rst

    ~pointcloud.base.GeometryPointCloudBase.pointcloud_equal
    ~pointcloud.base.PointCloudBase.georeferenced_coords_equal
```

(api-statistics)=
## Statistics

Most statistical functions are available directly from the objects and accessors, as shown above.

Below, we list functions defining additional estimators, variogram objects, and that can help plot.
### GeoUtils estimators

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    stats.nmad
    stats.linear_error
    stats.rmse
    stats.sum_square
```

### Variogram

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: variogram.rst

    Variogram
```

### Plotting helpers

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    stats.plot_grouped_stats
    Variogram.plot
```

## Numerical operators and uncertainty

See {ref}`core-operators` for neighborhoods, weights and custom definitions, and {ref}`uncertainty` for
estimation and propagation workflows.

### Operator interfaces and neighborhoods

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    operators.Interpolator
    operators.Reducer
    operators.LocalData
    operators.LinearCoefficients
    operators.PointNeighbours
    operators.GridNeighbours
```

### Interpolators

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    operators.interpolator.Nearest
    operators.interpolator.ScipyInterpolator
    operators.interpolator.Linear
    operators.interpolator.Cubic
    operators.interpolator.InverseDistance
    operators.interpolator.RasterConvolution
    operators.interpolator.Kriging
```

### Reducers

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    operators.reducer.Mean
    operators.reducer.Sum
    operators.reducer.Quantile
    operators.reducer.Median
    operators.reducer.Mode
    operators.reducer.Minimum
    operators.reducer.Maximum
    operators.reducer.Range
    operators.reducer.Count
    operators.reducer.RootMeanSquare
    operators.reducer.StandardDeviation
    operators.reducer.AverageDistance
    operators.reducer.AveragePairwiseDistance
```

### Error models and propagation

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    ErrorMagnitude
    ErrorComponent
    ErrorStructure
    ErrorStructure.estimate
    PropagationSummary
    uncertainty.propagate
    uncertainty.random_field
```

## Multiprocessing

Pass a {class}`~geoutils.multiproc.MultiprocConfig` as `mp_config` to supported methods for **chunked execution**
on GeoUtils objects. See {ref}`multiprocessing` for configuration and custom block examples.

### Configuration and clusters

Use {class}`~geoutils.multiproc.ClusterGenerator` to choose a {class}`~geoutils.multiproc.BasicCluster` (sequential)
or {class}`~geoutils.multiproc.MpCluster` (parallel).

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    multiproc.MultiprocConfig
    multiproc.ClusterGenerator
    multiproc.BasicCluster
    multiproc.MpCluster
```

### Block functions

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    multiproc.map_overlap
    multiproc.map_collect
```

(profiling-api)=
## Profiling

Measure function runtime and memory, or collect decorated workflow calls. See {ref}`profiling` for examples.

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: module.rst

    profiler

.. autosummary::

    profiler.profile_call
    profiler.profile
    profiler.ProfileMetrics
    profiler.Profiler
```
