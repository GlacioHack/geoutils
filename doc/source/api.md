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

RasterAccessor <api_rst>
Raster <api_raster>
VectorAccessor <api_vct>
Vector <api_vector>
PointCloudAccessor <api_pc>
PointCloud <api_pointcloud>
Inherited Vector methods <api_vector_inherited>
```

(raster-api)=
## Raster API

GeoUtils exposes the raster API through two mirrored interfaces:

- A {class}`rst <geoutils.RasterAccessor>` accessor extending {class}`xarray.DataArray` objects as rasters.
- {class}`~geoutils.Raster`, an interface operating directly on a GeoUtils object.


Both expose the **same methods and attributes**.

Only **file opening** and **scalable execution** differ between the two interfaces:

- **File opening:** {class}`~geoutils.Raster` objects are opened by instantiating the class, whereas {meth}`~geoutils.open_raster` is used for an {class}`xarray.DataArray` object,
- **Scalable execution:** the {class}`rst <geoutils.RasterAccessor>` accessor supports **Dask**, while the {class}`~geoutils.Raster` supports **Multiprocessing** instead.

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
    ~raster.base.RasterBase.interp_points
    ~raster.base.RasterBase.reduce_points
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

Writing and conversion methods depend on the concrete interface.

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Raster.to_file
    RasterAccessor.to_file
    Raster.to_rio_dataset
    Raster.to_xarray
    RasterAccessor.to_geoutils
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
## Vector

GeoUtils exposes the vector API through two mirrored interfaces:

- A {class}`vct <geoutils.VectorAccessor>` accessor extending {class}`geopandas.GeoDataFrame` objects as vectors.
- {class}`~geoutils.Vector`, an interface operating directly on a GeoUtils object.

Both expose the same GeoUtils-specific methods and attributes listed below. Native GeoPandas methods remain available
directly on the GeoDataFrame.

```{eval-rst}
.. minigallery:: geoutils.Vector geoutils.VectorAccessor
      :add-heading:
```

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
    :template: vector_method.rst

    ~vector.base.VectorBase.ds
    ~vector.base.VectorBase.crs
    ~vector.base.VectorBase.bbox
    ~vector.base.VectorBase.footprint
    ~vector.base.VectorBase.name
```

```{caution}
The {attr}`~VectorBase.bbox` attribute of a {class}`~geoutils.Vector` corresponds to the {attr}`~geopandas.GeoDataFrame.total_bounds` attribute of a
{class}`~geopandas.GeoDataFrame`, for consistency between rasters and vectors (and can also be accessed through {attr}`~VectorBase.total_bounds`).

The equivalent of {attr}`geopandas.GeoDataFrame.bounds` (i.e., a per-feature bounds) for {class}`Vectors<geoutils.Vector>` is {attr}`~geoutils.Vector.geom_bounds`.
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
## Point cloud

GeoUtils exposes the point cloud API through two mirrored interfaces:

- A {class}`pc <geoutils.PointCloudAccessor>` accessor extending {class}`geopandas.GeoDataFrame` objects as point clouds.
- {class}`~geoutils.PointCloud`, an interface operating directly on a GeoUtils object.

Both expose the same GeoUtils-specific methods and attributes listed below. Native GeoPandas methods remain available
directly on the GeoDataFrame.

```{eval-rst}
.. minigallery:: geoutils.PointCloud geoutils.PointCloudAccessor
      :add-heading:
```

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

    ~pointcloud.base.PointCloudBase.info
```

### Main attributes

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: pointcloud_method.rst

    ~pointcloud.base.PointCloudBase.ds
    ~pointcloud.base.PointCloudBase.data_column
    ~pointcloud.base.PointCloudBase.data
    ~pointcloud.base.PointCloudBase.crs
    ~pointcloud.base.PointCloudBase.bbox
    ~pointcloud.base.PointCloudBase.footprint
```

### Other attributes

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: pointcloud_method.rst

    ~pointcloud.base.PointCloudBase.point_count
```


### Create and convert from data

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: pointcloud_method.rst

    ~pointcloud.base.PointCloudBase.from_xyz
    ~pointcloud.base.PointCloudBase.from_array
    ~pointcloud.base.PointCloudBase.from_tuples
    ~pointcloud.base.PointCloudBase.to_xyz
    ~pointcloud.base.PointCloudBase.to_array
    ~pointcloud.base.PointCloudBase.to_tuples

```

### Geospatial

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: pointcloud_method.rst

    ~pointcloud.base.PointCloudBase.crop
    ~pointcloud.base.PointCloudBase.clip
    ~pointcloud.base.PointCloudBase.reproject
    ~pointcloud.base.PointCloudBase.translate
```

### Interface with raster

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: pointcloud_method.rst

    ~pointcloud.base.PointCloudBase.grid
```

### Plotting

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    :template: pointcloud_method.rst

    ~pointcloud.base.PointCloudBase.plot
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

    ~pointcloud.base.PointCloudBase.pointcloud_equal
    ~pointcloud.base.PointCloudBase.georeferenced_coords_equal
```

(api-statistics)=
## Statistics

The {ref}`Statistics feature page<stats>` introduces these functions and result objects. The corresponding spatial
methods are listed under {ref}`api-raster-statistics` and {ref}`api-point-statistics`.

### Array estimators

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    stats.nmad
    stats.linear_error
    stats.rmse
    stats.sum_square
```

### Grouped statistics and plotting

**Zonal statistics are grouped statistics with bins defined by vector features.** Use
`raster.stats(by={"zone": (zones, "id")})`, or the same point cloud method, to calculate statistics by
feature ID. See {ref}`stats-zonal` for examples. The array function below handles already aligned values and groupers.

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    stats.stats
    stats.plot_grouped_stats
```

### Variograms and covariance models

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    stats.variogram
    Variogram
```

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Variogram.estimate
    Variogram.from_pairs
    Variogram.from_model
    Variogram.fit
    Variogram.plot
    Variogram.variogram
    Variogram.covariance
    Variogram.correlation
    Variogram.combine
```

### Export and backend conversion

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Variogram.to_dataframe
    Variogram.to_xarray
    Variogram.to_dict
    Variogram.from_dict
    Variogram.from_skgstat
    Variogram.without_backend
    Variogram.to_gstools
    Variogram.to_gpytorch
```

(api-sampling)=
## Sampling

The {ref}`Sampling feature page<sampling>` describes selection from one dataset, matching locations between datasets,
and spatial pairs. Their object methods are listed under {ref}`api-raster-sampling` and {ref}`api-point-sampling`.

### Common-location samples

{meth}`~RasterBase.cosample` and
{meth}`~PointCloudBase.cosample` return a raster or point cloud on the
support selected by `at`. Accessor calls return an {class}`xarray.DataArray` or {class}`geopandas.GeoDataFrame`.
Bands or columns contain `"self"`, `"other"`, then named auxiliaries in mapping order. Raster band names are stored
in `tags["long_name"]` (Xarray `attrs["long_name"]`); point outputs retain the support's selected index labels and
use `"self"` as their active data column. See {ref}`sampling-cosample` for examples.

### Pair samples

{meth}`~RasterBase.pairsample` and
{meth}`~PointCloudBase.pairsample` return an {class}`xarray.Dataset`
with dimensions `pair` and `endpoint`, containing values, source indexes, coordinates and distances. See
{ref}`sampling-pairs` for its use and the available distance sampling schemes.

## Multiprocessing configuration

To perform **chunked execution** on GeoUtils objects, pass this Multiprocessing configuration to function that support it.

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    multiproc.MultiprocConfig
```
