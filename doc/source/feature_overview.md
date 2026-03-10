(feature-overview)=
# Feature and scalability overview

GeoUtils provides a unified API for manipulating **raster**, **vector**, and **point-cloud** data, and provides **scalable CPU execution** for most raster operations through Dask and Multiprocessing.

As many of our numerical operations rely on **NumPy, SciPy or Numba**, those are planned to be linked to their **GPU** counterparts (**CuPy** and **Numba CUDA**) in the future.

The **{ref}`summary tables<tables-overview>` directly below** lists the core features of GeoUtils, their scalability and available backends.
Further below, a series of **{ref}`illustrated examples<examples-overview>`** demonstrate these features. 

```{seealso}
If you are interested in porting from GDAL/OGR, see our {ref}`cheatsheet-osgeo` page.
While tables below provide a scalability summary, the detailed **input/output behaviour of all operations** is available on the {ref}`scalability-support` page.
```

## Summary

GeoUtils exposes a **consistent API across raster, vector and point-cloud objects** where possible (similar in spirit to the recent [GDAL CLI overhaul](https://gdal.org/en/stable/programs/index.html)). Many operations also support convenient **match-reference arguments** (e.g., matching a grid for reprojection or rasterization, bounds for cropping, or point coordinates for interpolation). See the {ref}`core-match-ref` page for details.

At its core, GeoUtils provides two interchangeable ways to work with geospatial data, exposing **identical APIs**:

- **Accessors** that extend existing data structures ({class}`rst <geoutils.RasterAccessor>` for **rasters** with **Xarray**, `pc` and `vct` for **point clouds** and **vectors** with **GeoPandas**),
- **GeoUtils objects** {class}`~geoutils.Raster`, {class}`~geoutils.PointCloud`, {class}`~geoutils.Vector`.

Nearly all **raster operations** support **scalable execution** using [Dask](https://www.dask.org/) or Multiprocessing, allowing large datasets to be processed **chunk-by-chunk without loading the full array into memory**. Support for **point-cloud operations** is partial and ongoing, while **vector operations** may gain scalable support in the future.

Additionally, some numerical routines of GeoUtils provide multiple computational **backends** (e.g., SciPy or Numba implementations).

All methods are tested to ensure they produce **identical results** whether executed **in-memory**, **chunked**, or with **different computational backends**.
(tables-overview)=
## Data operations

We first describe GeoUtils' core **data operations**, which operate on underlying arrays or geometries and can therefore benefit from **scalable execution**.

**Legend:** **“/”** indicates methods **shared across object types**, while **“⟷”** indicates methods **interfacing between two object types**.

```{list-table} Common API for data operations
:widths: 3 5 1 2
:header-rows: 1
:align: left
:class: tight-table

* - Method
  - Notes
  - Scalable
  - Backend

* - <span class="gu-table-section">Raster / Vector / Point</span>
  -
  -
  -

* - {meth}`~geoutils.Raster.reproject()`
  - Reproject to other CRS. Default tolerance parameters ensure chunk-invariance.
  - ✅
  - Rasterio / PyProj

* - {meth}`~geoutils.Raster.crop()`
  - Crop to bounds, either intersecting (untouched) allowing efficient I/O, or clipped (data modified).
  - ✅
  - Rasterio / GeoPandas

* - {meth}`~geoutils.Raster.translate()`
  - Apply a grid shift to object.
  - ✅
  - NumPy / GeoPandas
  
* - {meth}`~geoutils.Raster.plot()`
  - Visualization helper.
  - ❌
  - Matplotlib

* - <span class="gu-table-section">Raster / Point</span>
  -
  -
  -
  
* - {meth}`~geoutils.Vector.create_mask()`
  - Create boolean mask of a vector geometries over raster or point.
  - ✅
  - Rasterio / GeoPandas

* - {meth}`~geoutils.Raster.get_stats()`
  - Compute statistics of valid values over a valid mask.
  - ❌
  - NumPy / SciPy
  
* - {meth}`~geoutils.Raster.subsample()`
  - Randomly sample valid values. Chunk-invariant seed ensures reproducibility.
  - ✅
  - NumPy
  
* - {meth}`~geoutils.Raster.filter()`
  - Filter over window. Fast vectorized logic with NaN support.
  - ✅
  - SciPy
  
* - {meth}`~geoutils.Raster.proximity()`
  - Estimate proximity distance to target values.
  - ❌
  - SciPy
  

* - <span class="gu-table-section">Raster ⟷ Vector</span>
  -
  -
  -

* - {meth}`~geoutils.Raster.polygonize()`
  - Convert raster regions to vector polygons. Multiple chunked strategies for performance.
  - ✅
  - Rasterio / GeoPandas

* - {meth}`~geoutils.Vector.rasterize()`
  - Burn vector geometries onto a raster grid.
  - ✅
  - Rasterio

* - <span class="gu-table-section">Raster ⟷ Point</span>
  -
  -
  -

* - {meth}`~geoutils.Raster.interp_points()`
  - Interpolate raster at point locations. Fast regular-grid logic with added NaN propagation.
  - ✅
  - SciPy

* - {meth}`~geoutils.Raster.reduce_points()`
  - Aggregate raster values around points.
  - ❌
  - NumPy

* - {meth}`~geoutils.PointCloud.grid()`
  - Grid irregular points onto a raster grid. Multiple approaches with added NaN propagation. 
  - ❌
  - SciPy

* - {meth}`~geoutils.Raster.from_pointcloud_regular()`
  - Direct conversion when points lie on a regular grid.
  - ❌
  - NumPy
  
* - {meth}`~geoutils.Raster.to_pointcloud()`
  - Conversion to point cloud.
  - ❌
  - NumPy
```

## Metadata properties and operations

In addition to data operations, GeoUtils exposes **metadata** properties and methods consistently across geospatial objects. 
These operate only on metadata and therefore **do not load or modify underlying data arrays**.

```{list-table} Common API from metadata operations
:widths: 3 7
:header-rows: 1
:align: left
:class: tight-table

* - Attribute / Method
  - Description

* - <span class="gu-table-section">Raster / Vector / Point</span>
  -

* - {attr}`~geoutils.Raster.crs`
  - Coordinate reference system (CRS) of object.

* - {attr}`~geoutils.Raster.bounds`
  - Bounding box of object.

* - {attr}`~geoutils.Raster.footprint`
  - Footprint polygon geometry of object.
  
* - {attr}`~geoutils.Raster.is_loaded`
  - Whether geospatial object is loaded in-memory.

* - {attr}`~geoutils.Raster.name`
  - Filename of object on disk, if it exists.
  
* - {meth}`~geoutils.Raster.get_bounds_projected()`
  - Bounds projected in other CRS.

* - {meth}`~geoutils.Raster.get_footprint_projected()`
  - Footprint polygon geometry in other CRS.
 
* - {meth}`~geoutils.Raster.get_metric_crs()`
  - Get metric CRS suitable for this object.
  
* - {meth}`~geoutils.Raster.info()`
  - Summary of attributes for geospatial object.
  
* - <span class="gu-table-section">Raster / Point</span>
  -

* - {attr}`~geoutils.Raster.data`
  - Data array (2D grid for raster, 1D for point cloud).

* - {attr}`~geoutils.Raster.shape`
  - Shape of data array.

* - {attr}`~geoutils.Raster.is_mask`
  - Whether object is a mask. Clarifies ambiguity of raster/point file types often not supporting boolean types.

* - <span class="gu-table-section">Raster</span>
  -

* - {attr}`~geoutils.Raster.transform`
  - Geotransform to map raster indices to spatial coordinates.

* - {attr}`~geoutils.Raster.nodata`
  - Nodata value used to represent missing data on disk.
  
* - {attr}`~geoutils.Raster.area_or_point`
  - Pixel interpretation of raster values, either center point or area average.

* - <span class="gu-table-section">Point</span>
  -

* - {attr}`~geoutils.PointCloud.point_count`
  - Number of points in the point cloud.
```

[//]: # (&#40;examples-overview&#41;=)

[//]: # (## Examples)

[//]: # ()
[//]: # (The following presents a descriptive example show-casing core features of GeoUtils.)

[//]: # ()
[//]: # (```{code-cell} ipython3)

[//]: # (:tags: [remove-cell])

[//]: # ()
[//]: # (# To get a good resolution for displayed figures)

[//]: # (from matplotlib import pyplot)

[//]: # (pyplot.rcParams['figure.dpi'] = 600)

[//]: # (pyplot.rcParams['savefig.dpi'] = 600)

[//]: # (pyplot.rcParams['font.size'] = 9)

[//]: # (```)

[//]: # ()
[//]: # (### Core objects and accessors)

[//]: # ()
[//]: # (GeoUtils operates on **rasters, vectors and point clouds**.)

[//]: # ()
[//]: # (Below, we demonstrate features through **accessors**: {class}`rst <geoutils.RasterAccessor>` for **Xarray** rasters, `vct` for **GeoPandas** vectors, `pc` for **GeoPandas** point clouds.)

[//]: # (The exact same operations are available through the {class}`~geoutils.Raster`, {class}`~geoutils.Vector` and {class}`~geoutils.PointCloud` objects.)

[//]: # ()
[//]: # (**Accessors** operations always return another **Xarray** or **GeoPandas** object, while **GeoUtils object** operations always return a **GeoUtils object**.)

[//]: # (However, you can pass **any object** as an argument to a function.)

[//]: # ()
[//]: # (First, we open datasets with {meth}`~geoutils.open_raster` and {meth}`~geoutils.open_vector`, passing a `chunks` argument to trigger out-of-memory **Dask** behaviour for the raster.)

[//]: # ()
[//]: # (```{code-cell} ipython3)

[//]: # (import geoutils as gu)

[//]: # (import geopandas as gpd)

[//]: # ()
[//]: # (# Example files: infrared band of Landsat and glacier outlines)

[//]: # (filename_rast = gu.examples.get_path&#40;"everest_landsat_b4"&#41;)

[//]: # (filename_vect = gu.examples.get_path&#40;"everest_rgi_outlines"&#41;)

[//]: # ()
[//]: # (# Open datasets)

[//]: # (ds = gu.open_raster&#40;filename_rast, chunks={"x": 200, "y": 200}&#41;)

[//]: # (gdf = gpd.read_file&#40;filename_vect&#41;)

[//]: # ()
[//]: # (# Raster and vector objects)

[//]: # (ds)

[//]: # (gdf)

[//]: # (```)

[//]: # ()
[//]: # (GeoUtils accessors expose spatial methods directly on these objects while keeping compatibility with the broader **NumPy, Xarray and GeoPandas ecosystem**.)

[//]: # ()
[//]: # (### Plotting)

[//]: # ()
[//]: # (For plotting, GeoUtils includes lightweight plotting helpers to simplify common tasks such as overlaying rasters and vectors with multiple colorbars, while passing through standard plotting arguments.)

[//]: # ()
[//]: # (```{code-cell} ipython3)

[//]: # (# Plot raster and vector)

[//]: # (ds.rst.plot&#40;cbar_title="Distance to glacier outline"&#41;)

[//]: # (gdf.plot&#40;rast_proximity_to_vec, fc="none"&#41;)

[//]: # (```)

[//]: # ()
[//]: # (```{seealso})

[//]: # (For publication-quality cartographic figures, see)

[//]: # (Cartopy &#40;https://scitools.org.uk/cartopy/docs/latest/&#41; or)

[//]: # (GeoPlot &#40;https://residentmario.github.io/geoplot/index.html&#41;.)

[//]: # (```)

[//]: # ()
[//]: # (### Referencing, transformations and interfacing)

[//]: # ()
[//]: # (Most transformations and interfacing methods can accept another dataset as a **reference to match** during the operation &#40;for example matching bounds, CRS or grid resolution&#41;. See the {ref}`core-match-ref` page for details.)

[//]: # ()
[//]: # (Here, we crop the vector to the raster extent.)

[//]: # ()
[//]: # (```{code-cell} ipython3)

[//]: # (# Print initial bounds)

[//]: # (print&#40;gdf.vct.bounds&#41;)

[//]: # ()
[//]: # (# Crop vector to raster extent)

[//]: # (gdf_crop = gdf.vct.crop&#40;ds, clip=True&#41;)

[//]: # ()
[//]: # (print&#40;gdf_crop.bounds&#41;)

[//]: # (```)

[//]: # ()
[//]: # (As GeoUtils uses **consistent method names across object types** whenever possible, the same {meth}`~geoutils.Raster.bounds` attribute  )

[//]: # (and {meth}`~geoutils.Raster.crop` method would also work on a **raster** or **point cloud**.)

[//]: # ()
[//]: # (### Numerical operations and NumPy interface)

[//]: # ()
[//]: # (GeoUtils integrates naturally with the **scientific Python stack**.)

[//]: # ()
[//]: # (Objects, whether {class}`~geoutils.Raster` and {class}`~geoutils.PointCloud` or **Xarray/GeoPandas** data structures, can be used directly with **NumPy or SciPy**. )

[//]: # (Standard Python operations such as arithmetic, indexing or logical comparisons behave as expected.)

[//]: # ()
[//]: # (```{code-cell} ipython3)

[//]: # (import numpy as np)

[//]: # ()
[//]: # (# Normalize raster values)

[//]: # (ds = &#40;ds - np.min&#40;ds&#41;&#41; / &#40;np.max&#40;ds&#41; - np.min&#40;ds&#41;&#41;)

[//]: # (```)

[//]: # ()
[//]: # (### Masking and implicit behaviour)

[//]: # ()
[//]: # (Logical operations automatically create **boolean raster masks**:)

[//]: # ()
[//]: # (```{code-cell} ipython3)

[//]: # (mask_aoi = np.logical_and&#40;ds > 0.6, rast_proximity_to_vec > 200&#41;)

[//]: # (```)

[//]: # ()
[//]: # (These masks can be used for indexing, or to trigger implicit behaviour in certain spatial operations such as proximity calculation or polygonization where target pixels are clearly defined.)

[//]: # ()
[//]: # (```{code-cell} ipython3)

[//]: # (vect_aoi = mask_aoi.rst.polygonize&#40;&#41;)

[//]: # (```)

[//]: # ()
[//]: # (Finally, GeoUtils contains its own {attr}`is_mask` attribute and option on file opening to manipulate masks conveniently under-the-hood &#40;opening, reading, writing&#41;, as many geospatial file formats )

[//]: # (do not support **boolean types** and thus expect conversion to an integer type.)

[//]: # ()
[//]: # (### Higher-level analysis tools)

[//]: # ()
[//]: # (GeoUtils also provides higher-level spatial analysis tools.)

[//]: # ()
[//]: # (For example, one can compute the distance to vector geometries on a raster grid:)

[//]: # ()
[//]: # (```{code-cell} ipython3)

[//]: # (# Compute proximity to vector on raster grid)

[//]: # (rast_prox = vect.vct.proximity&#40;rast&#41;)

[//]: # (```)

[//]: # ()
[//]: # (```{note})

[//]: # (The raster array may still not be loaded in memory.)

[//]: # (GeoUtils loads data only when required by an operation.)

[//]: # (```)

[//]: # ()
[//]: # (## Saving results)

[//]: # ()
[//]: # (GeoUtils objects and derived datasets can easily be written to disk.)

[//]: # ()
[//]: # (```{code-cell} ipython3)

[//]: # (vect_aoi.to_file&#40;"myaoi.gpkg"&#41;)

[//]: # (```)



