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
(transformations)=
# Transformations

GeoUtils implements **transformations within and at the interface of** rasters, vectors and point clouds with **scalable execution**
for large datasets and consistent **resampling** techniques.

For all geospatial data objects, {ref}`transformations-same` are exposed through the same functions
(e.g. {meth}`reproject() <RasterBase.reproject>`). Operations between different object types are
grouped separately below in {ref}`transformation-rst-vct`, {ref}`transformation-rst-pc` and {ref}`transformations-mask`.

```{tip}
For convenience, all operations can be passed another raster, vector or point cloud as a **reference to match**.
In that case, no other argument is necessary. See {ref}`core-match-ref` for details.
```

```{code-cell} ipython3
:tags: [remove-cell]

# To get a good resolution for displayed figures
from matplotlib import pyplot
pyplot.rcParams['figure.dpi'] = 600
pyplot.rcParams['savefig.dpi'] = 600
pyplot.rcParams['font.size'] = 9
```

(transformations-same)=
## Same-type transformations

### Reproject

{meth}`ds.rst.reproject() or Raster.reproject() <RasterBase.reproject>`<br>
{meth}`gdf.vct.reproject() or Vector.reproject() <VectorBase.reproject>`<br>
{meth}`gdf.pc.reproject() or PointCloud.reproject() <PointCloudBase.reproject>`

Reprojection **transforms data from one coordinate reference system to another**.

For vectors and point clouds, the transformation of geometry points is **exact**. However, in the case of rasters, the projected points
do not necessarily fall on a regular grid and require **resampling**, which results in a slight loss of information.

For rasters, it can be useful to use {meth}`reproject() <RasterBase.reproject>` in the same CRS simply for re-gridding,
for instance when downsampling to a new resolution {attr}`~RasterBase.res`.

```{tip}
Due to the loss of information when re-gridding, it is important to **minimize the number of reprojections during the
analysis of rasters** (performing only one, if possible).

For the same reason, when comparing vector, point cloud and rasters in different CRSs, it is usually **better to reproject the vector or point cloud
 with no loss of information**, which is the default {ref}`sampling` behaviour of GeoUtils.
```

```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the code for opening example files"
:  code_prompt_hide: "Hide the code for opening example files"

import matplotlib.pyplot as plt
import geoutils as gu
ds = gu.open_raster(gu.examples.get_path("everest_landsat_b4"))
ds.rst.set_nodata(0)  # Annoying to have to do this here, should we update it in the example?
ds2 = gu.open_raster(gu.examples.get_path("everest_landsat_b4_cropped"))
gdf = gu.open_vector(gu.examples.get_path("everest_rgi_outlines"))
```

```{code-cell} ipython3
# Reproject vector to CRS of raster by simply passing the raster
gdf_reproj = gdf.vct.reproject(ds)
# Reproject raster to smaller bounds and different X/Y resolution
ds_reproj = ds.rst.reproject(
    res=(ds.rst.res[0] * 2, ds.rst.res[1] / 2),
    bounds={"left": ds.rst.bbox.left, "bottom": ds.rst.bbox.bottom,
            "right": ds.rst.bbox.left + 10000, "top": ds.rst.bbox.bottom + 10000},
    resampling="cubic")
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

f, ax = plt.subplots(1, 2)
ax[0].set_title("Before reprojection")
ds.rst.plot(ax=ax[0], cmap="gray", add_cbar=False)
gdf.vct.plot(ref=ds, ax=ax[0], ec="k", fc="none")
ax[1].set_title("After reprojection")
ds_reproj.rst.plot(ax=ax[1], cmap="gray", add_cbar=False)
gdf_reproj.vct.plot(ax=ax[1], ec="k", fc="none")
_ = ax[1].set_yticklabels([])
plt.tight_layout()
```

```{note}
In GeoUtils, `"bilinear"` is the default resampling method. A simple {class}`str` matching the naming of a {class}`rasterio.enums.Resampling` method can be
passed.

Resampling methods are listed in **[the dedicated section of Rasterio's API](https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling)**.
```

We can also simply pass another raster as reference to reproject to match the same CRS, and re-grid to the same bounds
and resolution:

```{code-cell} ipython3
---
mystnb:
  output_stderr: show
---
# Reproject raster to match another raster by simply passing it as reference
ds_reproj2 = ds.rst.reproject(ds2)
```

GeoUtils raises a warning because the rasters have different {ref}`Pixel interpretation<pixel-interpretation>`,
to ensure this is intended. This warning can be turned off at the package-level using GeoUtils' {ref}`config`.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

f, ax = plt.subplots(1, 3)
ax[0].set_title("Raster 1")
ds.rst.plot(ax=ax[0], cmap="gray", add_cbar=False)
gdf.vct.plot(ref=ds, ax=ax[0], ec="k", fc="none")
ax[1].set_title("Raster 2")
ds2.rst.plot(ax=ax[1], cmap="Reds", add_cbar=False)
gdf.vct.plot(ref=ds, ax=ax[1], ec="k", fc="none")
ax[2].set_title("Match-ref\nreprojection")
ds_reproj2.rst.plot(ax=ax[2], cmap="gray", add_cbar=False)
gdf_reproj.vct.plot(ax=ax[2], ec="k", fc="none")
_ = ax[1].set_yticklabels([])
_ = ax[2].set_yticklabels([])
plt.tight_layout()
```

### Crop

{meth}`ds.rst.crop() or Raster.crop() <RasterBase.crop>`<br>
{meth}`gdf.vct.crop() or Vector.crop() <VectorBase.crop>`<br>
{meth}`gdf.pc.crop() or PointCloud.crop() <PointCloudBase.crop>`

Cropping **selects data in a rectangular extent** without modifying the selected values or geometries. A file-backed
raster, vector or point cloud can store this selection and defer reading until its data are requested.

For rasters, crop removes complete rows and columns outside the extent. For point clouds, it removes points outside
the extent. For vectors, the default `mode="intersects"` keeps every unchanged feature touching the extent;
`mode="within"` keeps only features fully contained by it.

```{code-cell} ipython3
# Crop the vector to the smaller raster extent
gdf_cropped = gdf_reproj.vct.crop(ds2)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

f, ax = plt.subplots(1, 2)
ax[0].set_title("Before cropping")
ds2.rst.plot(ax=ax[0], cmap="gray", add_cbar=False)
gdf_reproj.vct.plot(ax=ax[0], ec="k", fc="none")
ax[1].set_title("After cropping")
ds2.rst.plot(ax=ax[1], cmap="gray", add_cbar=False)
gdf_cropped.vct.plot(ax=ax[1], ec="k", fc="none")
_ = ax[1].set_yticklabels([])
plt.tight_layout()
```

### Clip

{meth}`ds.rst.clip() or Raster.clip() <RasterBase.clip>`<br>
{meth}`gdf.vct.clip() or Vector.clip() <VectorBase.clip>`<br>
{meth}`gdf.pc.clip() or PointCloud.clip() <PointCloudBase.clip>`

Clipping **cuts values within exact geometry**. It masks raster cells outside the geometry, removes point
cloud rows outside it, and cuts vector geometries at its boundary. Raster, vector and point cloud clipping support
lazy Dask chunks or partitions and file-backed multiprocessing blocks.

```{code-cell} ipython3
# Clip the vector to the raster
gdf_clipped = gdf_reproj.vct.clip(ds)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

f, ax = plt.subplots(1, 2)
ax[0].set_title("Before clipping")
ds.rst.plot(ax=ax[0], cmap="gray", add_cbar=False)
gdf_reproj.vct.plot(ax=ax[0], ec="k", fc="none")
ax[1].set_title("After clipping")
ds.rst.plot(ax=ax[1], cmap="gray", add_cbar=False)
gdf_clipped.vct.plot(ax=ax[1], ec="k", fc="none")
_ = ax[1].set_yticklabels([])
plt.tight_layout()
```

### Translate

{meth}`ds.rst.translate() or Raster.translate() <RasterBase.translate>`<br>
{meth}`gdf.vct.translate() or Vector.translate() <VectorBase.translate>`<br>
{meth}`gdf.pc.translate() or PointCloud.translate() <PointCloudBase.translate>`

Translations **applies a horizontal offset to the georeferencing** without modifying the underlying data,
which is especially useful to align the data due to positioning errors.

```{code-cell} ipython3
# Translate the raster by a certain offset
ds_shift = ds.rst.translate(xoff=1000, yoff=1000)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

f, ax = plt.subplots(1, 2)
ax[0].set_title("Before translation")
ds.rst.plot(ax=ax[0], cmap="gray", add_cbar=False)
gdf_clipped.vct.plot(ax=ax[0], ec="k", fc="none")
ax[1].set_title("After translation")
ds_shift.rst.plot(ax=ax[1], cmap="gray", add_cbar=False)
gdf_clipped.vct.plot(ax=ax[1], ec="k", fc="none")
_ = ax[1].set_yticklabels([])
plt.tight_layout()
```

:::{admonition} See also
:class: tip

For 3D coregistration tailored to georeferenced elevation data, see [xDEM's coregistration module](https://xdem.readthedocs.io/en/stable/coregistration.html).
:::

### Merge

{func}`geoutils.raster.merge_rasters()`

Merge operations **join multiple geospatial data spatially, possibly with different georeferencing, into a single geospatial
data object**.

For rasters, the merging operation consists in combining all rasters into a single, larger raster. Pixels that overlap
are combined by a reductor function (defaults to the mean). The output georeferenced grid (CRS, transform and shape) can
be set to that of any reference raster (defaults to the extent that contains exactly all rasters).

This standalone function currently accepts {class}`~geoutils.Raster` objects.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for creating multiple raster pieces"
:  code_prompt_hide: "Show the code for creating multiple raster pieces"

# Get 4 cropped bits from initial rasters
rast = gu.Raster(gu.examples.get_path("everest_landsat_b4"))
rast1 = rast.crop((rast.bbox.left + 1000, rast.bbox.bottom + 1000,
                   rast.bbox.left + 3000, rast.bbox.bottom + 3000))
rast2 = rast.crop((rast.bbox.left + 3000, rast.bbox.bottom + 1000,
                   rast.bbox.left + 5000, rast.bbox.bottom + 3000))
rast3 = rast.crop((rast.bbox.left + 1000, rast.bbox.bottom + 3000,
                   rast.bbox.left + 3000, rast.bbox.bottom + 5000))
rast4 = rast.crop((rast.bbox.left + 3000, rast.bbox.bottom + 3000,
                   rast.bbox.left + 5000, rast.bbox.bottom + 5000))
# Reproject some in other CRS, with other resolution
#rast3 = rast3.reproject(crs=4326, res=rast.res[0] * 3)
#rast4 = rast4.reproject(crs=32610, res=rast.res[0] / 3)
```

```{code-cell} ipython3
---
mystnb:
  output_stderr: remove
---
# Merging all rasters, uses first raster's CRS, res, and the extent of all by default
merged_rast = gu.raster.merge_rasters([rast1, rast2, rast3, rast4])
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

f, ax = plt.subplots(1, 4)
ax[0].set_title("Raster 1")
rast1.plot(ax=ax[0], cmap="gray", add_cbar=False)
ax[1].set_title("Raster 2")
rast2.plot(ax=ax[1], cmap="gray", add_cbar=False)
ax[2].set_title("Raster 3")
rast3.plot(ax=ax[2], cmap="gray", add_cbar=False)
ax[3].set_title("Raster 4")
rast4.plot(ax=ax[3], cmap="gray", add_cbar=False)
plt.tight_layout()

merged_rast.plot(ax="new", cmap="gray", add_cbar=False)
```

(transformation-rst-vct)=
## Raster–vector transformations

### Rasterize

{meth}`gdf.vct.rasterize() or Vector.rasterize() <VectorBase.rasterize>`

Rasterization of a vector is **sets the values of raster pixels intersecting a vector geometry feature to that of an attribute**
(e.g., feature ID, area or any other value), which is the geometry index by default.

Rasterization generally implies some loss of information, as there is no exact way of representing a vector on a grid.
Rather, the choice of which pixels are attributed a value depends on the amount of intersection with the vector
geometries and so includes several options (percent of area intersected, all touched, etc).

```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the code for opening example files"
:  code_prompt_hide: "Hide the code for opening example files"

import matplotlib.pyplot as plt
import geoutils as gu
import numpy as np

ds = gu.open_raster(gu.examples.get_path("everest_landsat_b4"))
ds.rst.set_nodata(0)  # Annoying to have to do this here, should we update it in the example?
gdf = gu.open_vector(gu.examples.get_path("everest_rgi_outlines"))
```

```{code-cell} ipython3
# Rasterize the vector features based on their glacier ID number
rasterized_gdf = gdf.vct.rasterize(ds)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

f, ax = plt.subplots(1, 2)
ax[0].set_title("Raster and vector")
ds.rst.plot(ax=ax[0], cmap="gray", add_cbar=False)
gdf.vct.plot(ref=ds, ax=ax[0], ec="k", fc="none")
ax[1].set_title("Rasterized vector")
rasterized_gdf.rst.plot(ax=ax[1], cmap="viridis", cbar_title="Feature index")
_ = ax[1].set_yticklabels([])
plt.tight_layout()
```

### Polygonize

{meth}`ds.rst.polygonize() or Raster.polygonize() <RasterBase.polygonize>`

Polygonization of a raster **consists of delimiting contiguous raster pixels with the same target values into vector polygon
geometries**. By default, all raster values are used as targets. When using polygonize on a raster mask, i.e. a boolean {class}`~geoutils.Raster`,
the targets are implicitly the valid values of the mask.

```{code-cell} ipython3
# Mask 0 values
rasterized_gdf = rasterized_gdf.where(rasterized_gdf != 0)
# Polygonize all non-zero values
gdf_repolygonized = rasterized_gdf.rst.polygonize()

```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

f, ax = plt.subplots(1, 2)
ax[0].set_title("Raster (vector\n rasterized above)")
rasterized_gdf.rst.plot(ax=ax[0], cmap="viridis", cbar_title="Feature index")
ax[1].set_title("Polygonized raster")
gdf_repolygonized.vct.plot(ref=rasterized_gdf, ax=ax[1], column="id", fc="none", cbar_title="Feature index")
_ = ax[1].set_yticklabels([])
plt.tight_layout()
```

(transformation-rst-pc)=
## Raster–point transformations

### Interpolation at points

{meth}`ds.rst.interp_points() or Raster.interp_points() <RasterBase.interp_points>`

Point interpolation of a raster **consists in estimating the values at exact point coordinates by 2D regular-grid
interpolation** such as nearest neighbour, bilinear (default), cubic, etc.

```{note}
In order to support all types of resampling methods with nodata values while maintaining the robustness of results,
GeoUtils implements **a modified version of {func}`scipy.interpolate.interpn` that propagates nodata
values** in surrounding pixels of initial nodata values depending on the order of the resampling method:
- Nearest or linear (order 0 or 1): up to 1 pixel,
- Cubic (order 3): 2 pixels,
- Quintic (order 5): 3 pixels.
```

```{code-cell} ipython3
# We use a DEM, often requiring interpolation
ds = gu.open_raster(gu.examples.get_path("exploradores_aster_dem"))

# Get 50 random points to sample within the raster extent
rng = np.random.default_rng(42)
x_coords = rng.uniform(ds.rst.bbox.left, ds.rst.bbox.right, 50)
y_coords = rng.uniform(ds.rst.bbox.bottom, ds.rst.bbox.top, 50)

gdf_int = ds.rst.interp_points(points=(x_coords, y_coords))
```

The interpolated points can be returned as a {class}`~geopandas.GeoDataFrame` or
{class}`~geoutils.PointCloud`, enabling quick interfacing, or as an array.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

f, ax = plt.subplots(1, 2)
ax[0].set_title("Raster")
ds.rst.plot(ax=ax[0], cmap="terrain", cbar_title="Elevation (m)")
ax[1].set_title("Interpolated\npoint cloud")
gdf_int.pc.plot(ref=ds, ax=ax[1], cmap="terrain", cbar_title="Elevation (m)", marker="x")
_ = ax[1].set_yticklabels([])
plt.tight_layout()
```

### Reduction around point

Point reduction of a raster is **the estimation of the values at point coordinates by applying a reductor function (e.g., mean,
median) to pixels contained in a window centered on the point**. For a window smaller than the pixel size, the value of
the closest pixel is returned.

{meth}`ds.rst.reduce_points() or Raster.reduce_points() <RasterBase.reduce_points>`

```{code-cell} ipython3
gdf_red = ds.rst.reduce_points((x_coords, y_coords), window=5, reducer_function=np.nanmedian)
```

The reduced points can be returned as a {class}`~geopandas.GeoDataFrame` or {class}`~geoutils.PointCloud`, enabling
quick interfacing, or as an array.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

f, ax = plt.subplots(1, 2)
ax[0].set_title("Raster")
ds.rst.plot(ax=ax[0], cmap="terrain", cbar_title="Elevation (m)")
ax[1].set_title("Reduced\npoint cloud")
gdf_red.pc.plot(ref=ds, ax=ax[1], cmap="terrain", cbar_title="Elevation (m)")
_ = ax[1].set_yticklabels([])
plt.tight_layout()
```

### Raster to regular points

{meth}`ds.rst.to_pointcloud() or Raster.to_pointcloud() <RasterBase.to_pointcloud>`

**A raster can be converted exactly into a regular point cloud**, which each pixel in the raster is associated to its pixel
values to create a point cloud on a regular grid.
Optionally, it can also be subsampled at valid values only.

```{note}
For more details on subsampling, see the {ref}`sampling` feature page
```

```{code-cell} ipython3
points = ds.rst.to_pointcloud(subsample=10000)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

f, ax = plt.subplots(1, 2)
ax[0].set_title("Raster")
ds.rst.plot(ax=ax[0], cmap="terrain", cbar_title="Elevation (m)")
ax[1].set_title("Regular subsampled\npoint cloud")
points.pc.plot(ref=ds, ax=ax[1], cmap="terrain", cbar_title="Elevation (m)", markersize=2)
_ = ax[1].set_yticklabels([])
plt.tight_layout()
```

### Regular points to raster

{meth}`ds.rst.from_pointcloud_regular() or Raster.from_pointcloud_regular() <RasterBase.from_pointcloud_regular>`

**If a point cloud is regularly spaced in X and Y coordinates, it can be converted exactly into a raster**. Otherwise,
it must be re-gridded using {ref}`point-gridding` described below. For a regular point cloud, every point is associated to a
pixel in the raster grid, and the values are set to the raster. The point cloud does not necessarily need to contain
points for all grid coordinates, as pixels with no corresponding point are set to nodata values.

```{code-cell} ipython3
ds_from_points = ds.rst.from_pointcloud_regular(points, transform=ds.rst.transform, shape=ds.rst.shape)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

f, ax = plt.subplots(1, 2)
ax[0].set_title("Regular subsampled\npoint cloud")
points.pc.plot(ref=ds, ax=ax[0], cmap="terrain", legend=True, cbar_title="Elevation (m)", markersize=2)
ax[1].set_title("Raster from\npoint cloud")
ds_from_points.rst.plot(ax=ax[1], cmap="terrain", cbar_title="Elevation (m)")
_ = ax[1].set_yticklabels([])
plt.tight_layout()
```

(point-gridding)=
### Point gridding

Gridding of a point cloud **consists in estimating the values at 2D regular gridded coordinates based on an irregular
point cloud** using Delaunay triangular interpolation (default), inverse-distance weighting or circular statistics.

```{note}
For gridding, GeoUtils introduces nodata values in distances surrounding initial point coordinates, defaulting to a
distance of 1 pixel.
```

{meth}`gdf.pc.grid() or PointCloud.grid() <PointCloudBase.grid>`

```{code-cell} ipython3
# Grid the subsampled points back onto the raster reference
gridded_points = points.pc.grid(ds, dist_nodata_pixel=10)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

f, ax = plt.subplots(1, 2)
ax[0].set_title("Subsampled\npoint cloud")
points.pc.plot(ref=ds, ax=ax[0], cmap="terrain", cbar_title="Elevation (m)", markersize=2)
ax[1].set_title("Gridded raster\nfrom point cloud")
gridded_points.rst.plot(ax=ax[1], cmap="terrain", cbar_title="Elevation (m)")
_ = ax[1].set_yticklabels([])
plt.tight_layout()
```

(transformations-mask)=
## Masking from vector

{meth}`gdf.vct.create_mask() or Vector.create_mask() <VectorBase.create_mask>`

Mask creation from a vector classifies raster cells or point locations by whether they intersect any vector feature. It
therefore depends only on the geometries, independently of their attribute values.

```{code-cell} ipython3
# Open vector features matching the raster and interpolated points used above
gdf = gu.open_vector(gu.examples.get_path("exploradores_rgi_outlines"))

# Create a boolean raster mask from all vector features
raster_mask = gdf.vct.create_mask(ds)

# Create a boolean mask on the interpolated point cloud used above
point_mask = gdf.vct.create_mask(gdf_int)
```

With a raster reference, this returns a georeferenced boolean {class}`xarray.DataArray` for an accessor call, or a
{class}`~geoutils.Raster` mask for an object call. With a point cloud reference, it returns the same point coordinates
with a boolean active data column.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

# Define the colors used for the points and the False/True masks
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

mask_cmap = ListedColormap(["lightgray", "tab:blue"])
point_cmap = ListedColormap(["darkred"])

f, ax = plt.subplots(1, 3, figsize=(14, 5))
ax[0].set_title("Raster, vector\nand points")
ds.rst.plot(ax=ax[0], cmap="terrain", cbar_title="Raster elevation (m)")
gdf_int.pc.plot(ref=ds, ax=ax[0], cmap=point_cmap, add_cbar=False, marker="x", markersize=20)
gdf.vct.plot(ref=ds, ax=ax[0], ec="k", fc="none")
ax[0].legend(
    handles=[
        Line2D([0], [0], color="k", label="Vector contours"),
        Line2D([0], [0], color="darkred", marker="x", linestyle="none", label="Points"),
    ],
    loc="lower right",
)

ax[1].set_title("Raster\nmask")
_, raster_cax = raster_mask.rst.plot(
    ax=ax[1], cmap=mask_cmap, vmin=0, vmax=1, cbar_title="Intersects vector", return_axes=True
)
raster_cax.set_yticks([0, 1], labels=["False", "True"])
raster_cax.yaxis.labelpad = -8
gdf.vct.plot(ref=ds, ax=ax[1], ec="k", fc="none")

ax[2].set_title("Point cloud\nmask")
_, point_cax = point_mask.pc.plot(
    ref=ds, ax=ax[2], cmap=mask_cmap, vmin=0, vmax=1, cbar_title="Intersects vector", marker="x", return_axes=True
)
point_cax.set_yticks([0, 1], labels=["False", "True"])
point_cax.yaxis.labelpad = -8
gdf.vct.plot(ref=ds, ax=ax[2], ec="k", fc="none")

_ = ax[1].set_yticklabels([])
_ = ax[2].set_yticklabels([])
for axis in ax:
    axis.xaxis.set_major_locator(MaxNLocator(4))
plt.tight_layout()
```

Both outputs can be used as boolean selectors for their reference data.

```{code-cell} ipython3
# Mean of values in the mask
ds.where(raster_mask).mean().item()
```
