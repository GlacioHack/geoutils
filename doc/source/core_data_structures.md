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

# Data structures and interfaces

GeoUtils adds geospatial operations to familiar Xarray and GeoPandas data structures through accessors. It also
provides dedicated {class}`~geoutils.Raster`, {class}`~geoutils.Vector`, and {class}`~geoutils.PointCloud` objects
that expose the same GeoUtils methods.

| Data type | Xarray or GeoPandas structure | GeoUtils accessor | GeoUtils object |
|---|---|---|---|
| Raster | {class}`xarray.DataArray` | {class}`rst <geoutils.RasterAccessor>` | {class}`~geoutils.Raster` |
| Vector | {class}`geopandas.GeoDataFrame` | {class}`vct <geoutils.VectorAccessor>` | {class}`~geoutils.Vector` |
| Point cloud | {class}`geopandas.GeoDataFrame` | {class}`pc <geoutils.PointCloudAccessor>` | {class}`~geoutils.PointCloud` |

The accessors complement the methods already available from Xarray or GeoPandas. For example,
{meth}`~RasterBase.reproject` is a GeoUtils operation available through {class}`rst <geoutils.RasterAccessor>`, while
{meth}`xarray.DataArray.mean` remains an Xarray operation on the same raster data.

## GeoUtils objects

The dedicated objects use [class composition](https://realpython.com/inheritance-composition-python/#whats-composition)
to combine the data and geospatial metadata handled by NumPy, Rasterio, GeoPandas, and
[PyProj](https://pyproj4.github.io/pyproj/stable/index.html).

| GeoUtils object | Main data | Main geospatial metadata |
|---|---|---|
| {class}`~geoutils.Raster` | {class}`numpy.ma.MaskedArray` in {attr}`~RasterBase.data` | {attr}`~RasterBase.transform`, {attr}`~RasterBase.crs`, and {attr}`~RasterBase.nodata` |
| {class}`~geoutils.Vector` | {class}`geopandas.GeoDataFrame` in {attr}`~VectorBase.ds` | Geometry and {attr}`~VectorBase.crs` from the GeoDataFrame |
| {class}`~geoutils.PointCloud` | Point GeoDataFrame in {attr}`~VectorBase.ds`, with a main {attr}`~PointCloudBase.data_column` | Point geometry and {attr}`~VectorBase.crs` from the GeoDataFrame |

Derived attributes such as {attr}`~RasterBase.bbox`, {attr}`~RasterBase.res`, or {attr}`~VectorBase.bbox` describe
their georeferencing. File-backed objects can also keep the information needed to defer reading their data until an
operation requires it, as described in {ref}`core-lazy-load`.

GeoPandas methods are available directly from {class}`~geoutils.Vector` and {class}`~geoutils.PointCloud`. Their
outputs are cast according to their type: geometric outputs remain GeoUtils objects, while non-geometric outputs keep
their original GeoPandas or Pandas type.

## Choosing an interface

We recommend the {class}`rst <geoutils.RasterAccessor>`, {class}`vct <geoutils.VectorAccessor>`, and
{class}`pc <geoutils.PointCloudAccessor>` accessors for new workflows. They keep data in the Xarray and GeoPandas
structures used across the scientific Python ecosystem. The dedicated GeoUtils objects remain available for existing
workflows and provide the same GeoUtils-specific methods.

See the {ref}`raster-class`, {ref}`vector-class`, and {ref}`point-cloud` pages for detailed descriptions of each
object, or the {ref}`feature overview <feature-overview>` for operations shared across data types.
