(core-match-ref)=
# Match-reference operations

GeoUtils exposes convenient **match-reference** options across most of its operations.

Its objective is ease-of-use, by avoiding requiring volatile metadata extraction from users, potentially triggering unintended loading.

## What is match-reference ?

Match-reference means to **match** the georeferencing of a dataset to that of another **reference** dataset.

```{important}
A match-reference argument can be **any object** implementing the georeferencing attributes needed by the operation:
{attr}`~RasterBase.bbox`, {attr}`~RasterBase.crs` and/or {attr}`~RasterBase.transform`.
Matching a complete raster grid also requires {attr}`~RasterBase.shape` and an affine transform.
```

```{eval-rst}
.. plot:: diagrams/diagram_match_ref.py
    :width: 100%
    :alt: A source raster and reference produce a reprojected raster matching the reference grid, or a crop using the source grid within the reference bounding box.
```

Reprojection to a raster reference matches its **CRS and grid** (pixel size, alignment and bounding box). Cropping uses the
reference **bounding box**, preserving the source CRS and pixel size. Here the crop bounding box aligns with source cells;
otherwise its edges are rounded to the source grid.

## Matching with a raster, vector or point cloud

A vector or point cloud reference provides **a bounding box and CRS**. A raster reference also provides a **grid**:
its pixel size, alignment and shape.

**The same reference information serves the same purpose across object types.** Reprojection uses the reference CRS,
spatial selection uses its bounding box or geometry, and operations creating a raster grid can use its transform and shape.
An operation can use several of these together, as raster reprojection does with CRS and grid.


| Reference information | Operations using it |
|---|---|
| **CRS** | {meth}`raster.reproject() <RasterBase.reproject>`, {meth}`vector.reproject() <VectorBase.reproject>`, {meth}`point_cloud.reproject() <geoutils.pointcloud.base.GeometryPointCloudBase.reproject>` |
| **Bounding box and CRS** | {meth}`raster.crop() <RasterBase.crop>`, {meth}`vector.crop() <VectorBase.crop>`, {meth}`point_cloud.crop() <geoutils.pointcloud.base.GeometryPointCloudBase.crop>`; {meth}`raster.plot() <RasterBase.plot>`, {meth}`vector.plot() <VectorBase.plot>`, {meth}`point_cloud.plot() <geoutils.pointcloud.base.GeometryPointCloudBase.plot>`; {meth}`raster.intersection() <RasterBase.intersection>` for the overlap bounding box |
| **Geometry or footprint**, projected into the source CRS | {meth}`raster.clip() <RasterBase.clip>`, {meth}`vector.clip() <VectorBase.clip>`, {meth}`point_cloud.clip() <geoutils.pointcloud.base.GeometryPointCloudBase.clip>` |
| **Raster grid** (CRS, transform and shape) | {meth}`raster.reproject() <RasterBase.reproject>`, {meth}`raster.krige() <RasterBase.krige>`; {meth}`vector.rasterize() <VectorBase.rasterize>`, {meth}`vector.create_mask() <VectorBase.create_mask>`, {meth}`vector.proximity() <VectorBase.proximity>`; {meth}`point_cloud.grid() <PointCloudBase.grid>`, {meth}`point_cloud.krige() <PointCloudBase.krige>` |
| **Point locations** (X/Y, projected into the source CRS) | {meth}`raster.resample_at_points() <RasterBase.resample_at_points>`, {meth}`raster.interp_at_points() <RasterBase.interp_at_points>`, {meth}`raster.reduce_at_points() <RasterBase.reduce_at_points>`; {meth}`vector.create_mask() <VectorBase.create_mask>` with a point cloud reference |

When a reference supplies only a bounding box and CRS, rasterization and point gridding need `res` or `shape` to define the
grid. Raster reprojection instead estimates resolution from the source unless `res` or `grid_size` is given.
For point cloud gridding, reproject() the points first if their CRS differs from the reference.

Cropping selects existing cells, features or points using the reference bounding box, without changing their geometry.
Clipping follows the exact geometry boundary. Vector cropping selects whole intersecting features by default.
Raster intersection() aligns the overlap bounding box with the source grid by default (`match_ref=True`).

{func}`~geoutils.raster.stack_rasters` and {func}`~geoutils.raster.merge_rasters` use the reference **CRS and resolution**.
Their `reference` is a GeoUtils raster or input index (default: first input); the output bounding box covers the inputs' union unless
`use_ref_bounds=True` requests the reference bounding box.

Outputs derived from a raster often need no external reference: {meth}`~RasterBase.polygonize` and {meth}`~RasterBase.proximity`
inherit source georeferencing, and raster krige() uses the source grid unless a `ref` is supplied.
