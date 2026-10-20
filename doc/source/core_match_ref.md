(core-match-ref)=
# Match-reference operations

GeoUtils exposes convenient **match-reference** options across most of its operations.

Its objective is ease-of-use, by avoiding requiring volatile metadata extraction from users, potentially triggering unintended loading.

## What is match-reference ?

Match-reference means to **match** the georeferencing of a dataset to that of another **reference** dataset.

```{important}
A match-reference argument can be **any object** implementing the georeferencing attributes needed by the operation:
{attr}`~RasterBase.bbox`, {attr}`~RasterBase.crs` and/or {attr}`~RasterBase.transform`/{attr}`~RasterBase.shape`.
```

```{eval-rst}
.. plot:: diagrams/diagram_match_ref.py
    :width: 100%
    :alt: A source raster and reference produce a reprojected raster matching the reference grid, or a crop using the source grid within the reference bounding box.
```

Here, reprojection to a raster reference matches its **CRS and grid** (pixel size, alignment and bounding box). Cropping uses the
reference **bounding box**, preserving the source CRS and pixel size. 

## Matching with a raster, vector or point cloud

A vector or point cloud reference provides **a bounding box and CRS**. A raster reference also provides a **grid**:
its pixel size, alignment and shape.

**The same reference information serves the same purpose across object types.** Reprojection uses the reference CRS,
spatial selection uses its bounding box or geometry, and operations creating a raster grid can use its transform and shape.
An operation can use several of these together, as raster reprojection does with CRS and grid.


| Reference information | Operations using it                                                                                                                                                                                                                                                                                                                                                                                                           |
|---|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **CRS** | {meth}`raster.reproject() <RasterBase.reproject>`, {meth}`vector.reproject() <VectorBase.reproject>`, {meth}`point_cloud.reproject() <geoutils.pointcloud.base.GeometryPointCloudBase.reproject>`                                                                                                                                                                                                                             |
| **Bounding box and CRS** | {meth}`raster.crop() <RasterBase.crop>`, {meth}`vector.crop() <VectorBase.crop>`, {meth}`point_cloud.crop() <geoutils.pointcloud.base.GeometryPointCloudBase.crop>`; {meth}`raster.plot() <RasterBase.plot>`, {meth}`vector.plot() <VectorBase.plot>`, {meth}`point_cloud.plot() <geoutils.pointcloud.base.GeometryPointCloudBase.plot>` |
| **Geometry or footprint**, projected into the source CRS | {meth}`raster.clip() <RasterBase.clip>`, {meth}`vector.clip() <VectorBase.clip>`, {meth}`point_cloud.clip() <geoutils.pointcloud.base.GeometryPointCloudBase.clip>`                                                                                                                                                                                                                                                           |
| **Raster grid** (CRS, transform and shape) | {meth}`raster.reproject() <RasterBase.reproject>`, {meth}`raster.krige() <RasterBase.krige>`; {meth}`vector.rasterize() <VectorBase.rasterize>`, {meth}`vector.create_mask() <VectorBase.create_mask>` with a raster reference, {meth}`vector.proximity() <VectorBase.proximity>`; {meth}`point_cloud.grid() <PointCloudBase.grid>`, {meth}`point_cloud.krige() <PointCloudBase.krige>`                                       |
| **Point locations** (X/Y, projected into the source CRS) | {meth}`raster.resample_at_points() <RasterBase.resample_at_points>`, {meth}`raster.interp_at_points() <RasterBase.interp_at_points>`, {meth}`raster.reduce_at_points() <RasterBase.reduce_at_points>`; {meth}`vector.create_mask() <VectorBase.create_mask>` with a point cloud reference                                                                                                                                     |


**When only a bounding box and CRS are supplied to an operation requiring a grid definition**, such as rasterization, gridding or reprojection, 
then **one of resolution `res` or output shape `shape`** must be defined. Raster reprojection is an exception, as it can 
estimate resolution from the source instead. 

Note that **outputs derived from a raster often need no external reference**: for example, {meth}`~RasterBase.polygonize` 
and {meth}`~RasterBase.proximity` inherit georeferencing of the source raster directly.
