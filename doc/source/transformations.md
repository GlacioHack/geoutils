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

GeoUtils implements **transformations on, and at the interface of,** rasters, vectors and point clouds with **scalable execution**
for large datasets and consistent **resampling** techniques.

For all geospatial data objects, {ref}`transformations-same` are exposed through the same functions
(e.g. {meth}`reproject() <RasterBase.reproject>`). Operations between different object types are
grouped separately below in {ref}`transformation-rst-vct`, {ref}`transformation-rst-pc` and {ref}`transformations-mask`.

```{tip}
For convenience, many operations accept another raster, vector or point cloud as a **reference to match**.
See {ref}`core-match-ref` for details.
```

```{code-cell} ipython3
:tags: [remove-cell]

# To get a good resolution for displayed figures
from matplotlib import pyplot
pyplot.rcParams['figure.dpi'] = 600
pyplot.rcParams['savefig.dpi'] = 600
pyplot.rcParams['font.size'] = 9

# Show intentional warnings as just their type and message
import warnings
warnings.formatwarning = lambda message, category, *args, **kwargs: f"{category.__name__}: {message}\n"
```

(transformations-resampling)=
## Choosing a resampling method

Resampling is **shared among several transformations operations**, such as reprojection or point gridding.
It can use either interpolators (e.g. linear interpolation) or reducers (e.g. window average) to estimate values over a new spatial support.
Below, we summarize available methods and backends.

```{note}
In GeoUtils, `"linear"` is the default resampling method.

Refer to {ref}`core-operators` for operator, neighborhood and nodata customization.
```

(transformations-interpolators)=
### Interpolators

The following **interpolation methods and backends** are supported, for point and raster inputs:

```{eval-rst}
.. plot:: diagrams/diagram_resampling.py make_interpolator_diagram
    :width: 100%
```

Note that "linear" and "cubic" interpolators for point inputs first use **Delauney triangulation**.

(transformations-reducers)=
### Reducers

And the following **reduction methods and backends**:

```{eval-rst}
.. plot:: diagrams/diagram_resampling.py make_reducer_diagram
    :width: 100%
```

Note that some reducer **do not retain the unit of the input**.

```{tip}
Due to loss of information, it is important to **minimize the number of resampling during analysis** (performing only one, if possible).

For the same reason, when comparing vector, point cloud and rasters in different CRSs, it is usually **better to reproject the vector or point cloud
 with no loss of information**, which is the default {ref}`sampling-cosample` behaviour of GeoUtils.
```

(transformations-same)=
## Same-type transformations

### Reproject

{meth}`ds.rst.reproject() or Raster.reproject() <RasterBase.reproject>`<br>
{meth}`gdf.vct.reproject() or Vector.reproject() <VectorBase.reproject>`<br>
{meth}`gdf.pc.reproject() or PointCloud.reproject() <geoutils.pointcloud.base.GeometryPointCloudBase.reproject>`

Reprojection **transforms data from one coordinate reference system to another**.

For vectors and point clouds, the transformation of geometry points is **exact**. However, in the case of rasters, the projected points
do not necessarily fall on a regular grid and require **resampling**, which results in a slight loss of information.

For rasters, it can be useful to use {meth}`reproject() <RasterBase.reproject>` in the same CRS simply for re-gridding,
for instance when downsampling to a new resolution {attr}`~RasterBase.res`.

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

# Reproject vector to CRS of raster by simply passing the raster
gdf_reproj = gdf.vct.reproject(ds)

# Select one glacier fully covered by the raster for the crop example
crop_glacier = gdf.vct.crop(ds, mode="within").sort_values("Area", ascending=False).head(1)
```

You can define the destination grid with a **bounding box and resolution**. Here, we stay in the source CRS
and use 500 × 1000 m cells over a centered 10 × 10 km square.

```{code-cell} ipython3
# Reproject raster to smaller bounds and different X/Y resolution
center_x = (ds.rst.bbox.left + ds.rst.bbox.right) / 2
center_y = (ds.rst.bbox.bottom + ds.rst.bbox.top) / 2
ds_regridded = ds.rst.reproject(
    res=(500, 1000),
    bounds={"left": center_x - 5000, "bottom": center_y - 5000,
            "right": center_x + 5000, "top": center_y + 5000},
    resampling="cubic")
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

# Use the same value scale for the source and result
value_limits = {"vmin": float(ds.min()), "vmax": float(ds.max())}
f, ax = plt.subplots(1, 3, figsize=(11, 4))
ax[0].set_title(f"Source values\n{ds.rst.crs.to_string()}")
ds.rst.plot(ax=ax[0], cmap="Blues", add_cbar=False, **value_limits)
gdf.vct.plot(ref=ds, ax=ax[0], ec="darkorange", fc="none")

# Locate the destination grid on the original raster
left, bottom, right, top = ds_regridded.rst.bbox
ax[0].add_patch(Rectangle(
    (left, bottom), right - left, top - bottom,
    fill=False, edgecolor="red", linestyle="--", linewidth=2, label="Bounding box of\ndestination grid",
))
ax[0].legend(loc="upper right")
ax[0].set_ylabel("Northing (m)")

# Show the manually defined grid before and after resampling
ax[1].set_title(f"Manual grid (20 × 10)\n{ds_regridded.rst.crs.to_string()}")
ax[1].set_facecolor("lavender")
ax[2].set_title(f"Values on manual grid\n{ds_regridded.rst.crs.to_string()}")
ds_regridded.rst.plot(ax=ax[2], cmap="Blues", add_cbar=False, interpolation="nearest", **value_limits)
gdf.vct.plot(ref=ds_regridded, ax=ax[2], ec="darkorange", fc="none")

# Draw every cell edge with the requested X/Y spacing in both panels
x_edges = np.linspace(left, right, ds_regridded.rst.width + 1)
y_edges = np.linspace(bottom, top, ds_regridded.rst.height + 1)
for axis in ax[1:]:
    axis.vlines(x_edges, bottom, top, colors="purple", linewidth=0.4, alpha=0.6)
    axis.hlines(y_edges, left, right, colors="purple", linewidth=0.4, alpha=0.6)
    axis.set(xlim=(left, right), ylim=(bottom, top), aspect="equal")
ax[1].set_ylabel("Northing (m)")
ax[2].tick_params(labelleft=False)
for axis in ax:
    axis.set_xlabel("Easting (m)")
    axis.xaxis.set_major_locator(MaxNLocator(3))
plt.tight_layout()
```

```{note}
GeoUtils resampling uses `tolerance=0` by default to calculate exact pixel transformations and make results
consistent across chunks. Rasterio/GDAL's default is 0.125 pixel, which allows faster computation but can
change results depending on chunk bounds. Passing the `tolerance` option requires Rasterio 1.5 or newer.

When values represent elevations and the transformation should also change their vertical datum,
either use `apply_vertical=True` or use [xDEM](https://xdem.readthedocs.io/en/stable/)'s reprojection which accounts
for vertical referencing implicitly.
```

Passing another raster as a reference matches its **CRS, bounds, resolution and grid alignment**.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for creating the reference grid"
:  code_prompt_hide: "Hide the code for creating the reference grid"

import numpy as np
from rasterio.transform import from_bounds

# Create a geographic reference containing only missing values
reference_bounds = ds2.rst.get_bbox_projected(4326)
reference_transform = from_bounds(*reference_bounds, width=30, height=20)
reference = gu.raster.from_array(
    np.full((20, 30), np.nan), reference_transform, crs=4326, area_or_point="Area"
)
```

```{code-cell} ipython3
---
mystnb:
  output_stderr: show
---
# Reproject raster to match another raster by simply passing it as reference
ds_reproj = ds.rst.reproject(reference)
```

GeoUtils raises a warning because the rasters have different {ref}`Pixel interpretation<pixel-interpretation>`,
to ensure this is intended. This warning can be turned off at the package-level using GeoUtils' {ref}`config`.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

from matplotlib.ticker import FormatStrFormatter, MaxNLocator

# Use the same value scale for the source and result
value_limits = {"vmin": float(ds.min()), "vmax": float(ds.max())}
f, ax = plt.subplots(1, 3, figsize=(11, 4))
ax[0].set_title(f"Source values\n{ds.rst.crs.to_string()}")
ds.rst.plot(ax=ax[0], cmap="Blues", add_cbar=False, **value_limits)
gdf.vct.plot(ref=ds, ax=ax[0], ec="darkorange", fc="none")
ax[0].set_xlabel("Easting (m)")
ax[0].set_ylabel("Northing (m)")
ax[0].xaxis.set_major_locator(MaxNLocator(3))

# Locate the reference bounding box in the source CRS
reference_in_source = reference.rst.get_bbox_projected(ds.rst.crs)
ax[0].add_patch(Rectangle(
    (reference_in_source.left, reference_in_source.bottom),
    reference_in_source.right - reference_in_source.left,
    reference_in_source.top - reference_in_source.bottom,
    fill=False, edgecolor="red", linestyle="--", linewidth=2,
    label="Reference bounding box\nin source CRS",
))
ax[0].legend(loc="upper right", fontsize=8)

# Show the reference as cell boundaries, without plotting its values
ax[1].set_title(f"Reference grid (30 × 20)\n{reference.rst.crs.to_string()}")
ax[1].set_facecolor("lavender")
ax[2].set_title(f"Values on reference grid\n{ds_reproj.rst.crs.to_string()}")
ds_reproj.rst.plot(ax=ax[2], cmap="Blues", add_cbar=False, interpolation="nearest", **value_limits)
gdf.vct.plot(ref=reference, ax=ax[2], ec="darkorange", fc="none")

# Draw identical cell edges and map extents on the reference and result
left, bottom, right, top = reference.rst.bbox
x_edges = np.linspace(left, right, reference.rst.width + 1)
y_edges = np.linspace(bottom, top, reference.rst.height + 1)
latitude_aspect = 1 / np.cos(np.deg2rad((bottom + top) / 2))
for axis in ax[1:]:
    axis.vlines(x_edges, bottom, top, colors="purple", linewidth=0.4, alpha=0.6)
    axis.hlines(y_edges, left, right, colors="purple", linewidth=0.4, alpha=0.6)
    axis.set(xlim=(left, right), ylim=(bottom, top), aspect=latitude_aspect, xlabel="Longitude (°)")
    axis.xaxis.set_major_locator(MaxNLocator(3))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
ax[1].set_ylabel("Latitude (°)")
ax[2].tick_params(labelleft=False)
plt.tight_layout()
```

#### GCP/RPC reprojection

Rasters referenced by **ground control points (GCPs)** or **rational polynomial coefficients (RPCs)** can also be
reprojected, with chunked execution support through Dask/multiprocessing.
GeoUtils reads and uses these models from file automatically. The result has a regular affine grid. GCPs use their stored CRS, and RPCs describe WGS84 coordinates.
GCP/RPC reprojection requires a Rasterio resampling method, such as `"bilinear"`, which interpolates pixel values
onto the output grid, while GDAL transformer options control how GCPs/RPCs map pixel positions to ground coordinates
(see [Rasterio's documentation](https://rasterio.readthedocs.io/en/stable/topics/reproject.html#reprojecting-with-other-georeferencing-metadata)).

We give the ASTER DEM GCPs at its four corners. The `GCP_TPS` option uses thin plate splines to map pixels smoothly through the
control points.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for setting the GCPs"
:  code_prompt_hide: "Hide the code for setting the GCPs"

import rasterio as rio

# Open the ASTER DEM and use its corner coordinates as GCPs
source = gu.Raster(gu.examples.get_path("exploradores_aster_dem"))

# Set their height to zero
gcps = []
for row in (0, source.height):
    for col in (0, source.width):
        x, y = source.transform * (col, row)
        gcps.append(rio.control.GroundControlPoint(row=row, col=col, x=x, y=y, z=0))

# Assign the GCPs and their CRS to the DEM
source.gcps = (gcps, source.crs)
```

```{code-cell} ipython3
# Reproject using the assigned GCPs and thin plate splines
result = source.reproject(
    crs=4326, resampling="bilinear", transformer_options={"SRC_METHOD": "GCP_TPS"}
)

# Show the regular affine grid created by reprojection
print(result.crs)
print(result.transform)
```

### Crop

{meth}`ds.rst.crop() or Raster.crop() <RasterBase.crop>`<br>
{meth}`gdf.vct.crop() or Vector.crop() <VectorBase.crop>`<br>
{meth}`gdf.pc.crop() or PointCloud.crop() <geoutils.pointcloud.base.GeometryPointCloudBase.crop>`

Cropping **selects data in a rectangular extent** without modifying the selected values or geometries. A file-backed
raster, vector or point cloud can store this selection and defer reading until its data are requested.

For rasters, crop removes complete rows and columns outside the extent. For point clouds, it removes points outside
the extent. For vectors, the default `mode="intersects"` keeps every unchanged feature touching the extent;
`mode="within"` keeps only features fully contained by it.

First, we crop a raster to the **bounding box of a vector**:

```{code-cell} ipython3
# Crop the raster to the selected glacier's extent
ds_cropped = ds.rst.crop(crop_glacier)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

f, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].set_title("Before cropping")
ds.rst.plot(ax=ax[0], cmap="Blues", add_cbar=False)
crop_glacier.vct.plot(ref=ds, ax=ax[0], ec="darkorange", fc="none", lw=1.5)
ax[1].set_title("Raster cropped to vector bounds")
ds_cropped.rst.plot(ax=ax[1], cmap="Blues", add_cbar=False)
crop_glacier.vct.plot(ref=ds_cropped, ax=ax[1], ec="darkorange", fc="none", lw=1.5)
_ = ax[1].set_yticklabels([])
plt.tight_layout()
```

Then, for a vector, we crop **intersect or fully contained features** using the smaller raster as the crop extent:

```{code-cell} ipython3
# Select whole features using the same extent with either mode
gdf_intersects = gdf_reproj.vct.crop(ds2, mode="intersects")
gdf_within = gdf_reproj.vct.crop(ds2, mode="within")
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

from matplotlib.patches import Rectangle

f, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
crop_bounds = ds2.rst.bbox
for axis, selected, mode in zip(ax, [gdf_intersects, gdf_within], ["intersects", "within"]):
    axis.set_title(f'Crop vector with\nmode = "{mode}"')
    ds.rst.plot(ax=axis, cmap="Blues", add_cbar=False)
    selected.vct.plot(ref=ds, ax=axis, ec="darkorange", fc="none", lw=1.5)

    # Mark the requested extent without cutting the selected geometries
    axis.add_patch(Rectangle(
        (crop_bounds.left, crop_bounds.bottom),
        crop_bounds.right - crop_bounds.left, crop_bounds.top - crop_bounds.bottom,
        fill=False, edgecolor="purple", linestyle="--", linewidth=1.5, label="Crop extent",
    ))
    axis.legend(loc="lower left")
plt.tight_layout()
```

### Clip

{meth}`ds.rst.clip() or Raster.clip() <RasterBase.clip>`<br>
{meth}`gdf.vct.clip() or Vector.clip() <VectorBase.clip>`<br>
{meth}`gdf.pc.clip() or PointCloud.clip() <geoutils.pointcloud.base.GeometryPointCloudBase.clip>`

Clipping **cuts values within exact geometry**. It masks raster cells outside the geometry, removes point
cloud rows outside it, and cuts vector geometries at its boundary. Raster, vector and point cloud clipping support
lazy Dask chunks or partitions and file-backed multiprocessing blocks.

Below, we first clip a raster with an **exact vector geometry**, turning outside cells into NaNs.

```{code-cell} ipython3
# Clip the cropped raster to the glacier geometry
ds_clipped = ds_cropped.rst.clip(crop_glacier)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

from matplotlib.patches import Patch

# Compare the raster before and after clipping with the same extent and value scale
clip_value_limits = {"vmin": float(ds_cropped.min()), "vmax": float(ds_cropped.max())}
f, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
ax[0].set_title("Raster + exact\nvector geometry")
ds_cropped.rst.plot(ax=ax[0], cmap="Blues", add_cbar=False, **clip_value_limits)
ax[1].set_title("Raster clipped with\nexact vector geometry")
ds_clipped.rst.plot(ax=ax[1], cmap="Blues", add_cbar=False, **clip_value_limits)

# Show missing raster values in gray and outline the clipping geometry on both panels
for axis in ax:
    axis.set_facecolor("lightgray")
    crop_glacier.vct.plot(ref=ds_cropped, ax=axis, ec="darkorange", fc="none", lw=1.5)
    axis.xaxis.set_major_locator(MaxNLocator(3))
ax[1].legend(handles=[Patch(facecolor="lightgray", label="NaN")], loc="lower right")
ax[1].tick_params(labelleft=False)
plt.tight_layout()
```

Then, we clip a vector with a **raster footprint**, cutting its geometries at the raster edges:

```{code-cell} ipython3
# Clip the vector to the raster
gdf_clipped = gdf_reproj.vct.clip(ds)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

f, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].set_title("Vector + raster footprint")
ds.rst.plot(ax=ax[0], cmap="Blues", add_cbar=False)
gdf_reproj.vct.plot(ax=ax[0], ec="darkorange", fc="none")

# Show the full original vector extent in both panels
clip_xlim, clip_ylim = ax[0].get_xlim(), ax[0].get_ylim()
ax[1].set_title("Vector clipped with\nraster footprint")
ds.rst.plot(ax=ax[1], cmap="Blues", add_cbar=False)
gdf_clipped.vct.plot(ax=ax[1], ec="darkorange", fc="none")
ax[1].set_xlim(clip_xlim)
ax[1].set_ylim(clip_ylim)

# Space coordinate labels consistently across the shared map extent
from matplotlib.ticker import MaxNLocator
raster_bounds = ds.rst.bbox
ax[0].add_patch(Rectangle(
    (raster_bounds.left, raster_bounds.bottom),
    raster_bounds.right - raster_bounds.left, raster_bounds.top - raster_bounds.bottom,
    fill=False, edgecolor="red", linestyle="--", linewidth=1.5, label="Raster footprint",
))
for axis in ax:
    axis.xaxis.set_major_locator(MaxNLocator(4))
ax[0].legend(loc="upper right")
_ = ax[1].set_yticklabels([])
plt.tight_layout()
```

### Translate

{meth}`ds.rst.translate() or Raster.translate() <RasterBase.translate>`<br>
{meth}`gdf.vct.translate() or Vector.translate() <VectorBase.translate>`<br>
{meth}`gdf.pc.translate() or PointCloud.translate() <geoutils.pointcloud.base.GeometryPointCloudBase.translate>`

Translation **applies a horizontal offset to the georeferencing** without modifying the underlying data,
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
ds.rst.plot(ax=ax[0], cmap="Blues", add_cbar=False)
gdf_clipped.vct.plot(ref=ds, ax=ax[0], ec="darkorange", fc="none")
ax[1].set_title("After translation")
ds_shift.rst.plot(ax=ax[1], cmap="Blues", add_cbar=False)
gdf_clipped.vct.plot(ref=ds_shift, ax=ax[1], ec="darkorange", fc="none")
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

Rasterization **sets the values of raster pixels intersecting a vector geometry feature to that of an attribute**
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
ds.rst.plot(ax=ax[0], cmap="Blues", add_cbar=False)
gdf.vct.plot(ref=ds, ax=ax[0], ec="darkorange", fc="none")
ax[1].set_title("Rasterized vector")
rasterized_gdf.rst.plot(ax=ax[1], cmap="viridis", cbar_title="Feature index")
_ = ax[1].set_yticklabels([])
plt.tight_layout()
```

(transformations-fractional)=
### Fractional rasterization

Standard rasterization assigns a unique value to selected cells, while fractional rasterization instead returns the
**fraction of each cell covered by a geometry**, from 0 to 1. This is useful when polygon edges cut through cells
and whole-cell membership would distort the represented area.

```{code-cell} ipython3
# Calculate the fraction of each raster cell covered by the union of glacier polygons
glacier_fraction = gdf.vct.rasterize(ref=ds, fractional="union")
```
We zoom into a glacier margin to show **partial coverage of individual 30 m cells**:

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

# Compare the same seven by seven cells along a curved glacier margin
fraction_zoom = glacier_fraction.rst.crop((490390, 3098840, 490600, 3099050))
standard_zoom = gdf.vct.rasterize(ref=fraction_zoom, in_value=1)
left, bottom, right, top = fraction_zoom.rst.bbox
x_edges = np.linspace(left, right, fraction_zoom.rst.width + 1)
y_edges = np.linspace(bottom, top, fraction_zoom.rst.height + 1)
x_centers, y_centers = fraction_zoom.rst.coords(force_offset="center")

# Reserve a separate colorbar column so both maps have the same size
f, axes = plt.subplots(1, 3, figsize=(9, 4), gridspec_kw={"width_ratios": [1, 1, 0.05]})
for axis, coverage, title in zip(
    axes[:2], [standard_zoom, fraction_zoom], ["Standard (cell center)", "Fractional coverage"]
):
    coverage.rst.plot(ax=axis, cmap="Blues", vmin=0, vmax=1, add_cbar=False, interpolation="nearest")
    gdf.vct.plot(ref=fraction_zoom, ax=axis, ec="darkorange", fc="none", lw=2)
    axis.vlines(x_edges, bottom, top, colors="slategray", linewidth=0.6)
    axis.hlines(y_edges, left, right, colors="slategray", linewidth=0.6)

    # Label coverage fractions, distinguishing almost empty or full cells from exact 0 or 1
    for row, col in np.ndindex(coverage.shape):
        value = float(coverage.values[row, col])
        if np.isclose(value, 0) or np.isclose(value, 1):
            label = f"{value:.0f}"
        elif value < 0.01:
            label = "<0.01"
        elif value > 0.99:
            label = ">0.99"
        else:
            label = f"{value:.2f}"
        axis.text(
            x_centers[row, col], y_centers[row, col], label,
            ha="center", va="center", fontsize=9,
            color="white" if value > 0.6 else "black",
        )

    # Use identical bounds and coordinate labels in the two panels
    axis.set(title=title, xlim=(left, right), ylim=(bottom, top), xlabel="Easting (m)")
    axis.xaxis.set_major_locator(MaxNLocator(3))
    axis.yaxis.set_major_locator(MaxNLocator(3))
axes[0].set_ylabel("Northing (m)")
axes[1].tick_params(labelleft=False)
f.colorbar(axes[1].images[0], cax=axes[2], label="Fraction of cell covered", ticks=[0, 0.25, 0.5, 0.75, 1])
plt.tight_layout()
```

Using `fractional=True` returns one layer per feature, `fractional_by="column"` combines features sharing a label into
one layer per group; `fractional="union"` combines all features into one layer. Overlaps within a group count once,
while different groups can overlap independently.

See {ref}`core-coverage` for coverage semantics and {ref}`stats-fractional` for calculating zonal statistics directly, 
without storing fractional coverage rasters.

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

{meth}`ds.rst.interp_at_points() or Raster.interp_at_points() <RasterBase.interp_at_points>`

Point interpolation of a raster **consists in estimating the values at exact point coordinates by 2D regular-grid
interpolation** such as nearest neighbour, bilinear (default), cubic, etc.

See {ref}`interpolator comparisons and options <transformations-interpolators>` to choose the prediction method.

```{note}
The default `nodata_handling="nearest"` masks a prediction when its nearest source observation is missing.
Use `"ignore"` to calculate from valid observations or `"propagate"` to mask results affected by missing inputs.
An integer or `"half_order_up"` / `"half_order_down"` instead spreads the missing-data mask by a distance in
source pixels. See {ref}`core-nodata` and {ref}`core-custom-operators` for these choices.
```

Tuple coordinates are supplied in **X/Y order**. With `input_latlon=True`, use **longitude, latitude** in degrees;
a georeferenced point cloud supplies its own CRS and is projected automatically.

```{code-cell} ipython3
# We use a DEM, often requiring interpolation
ds = gu.open_raster(gu.examples.get_path("exploradores_aster_dem"))

# Get 50 random points to sample within the raster extent
rng = np.random.default_rng(42)
x_coords = rng.uniform(ds.rst.bbox.left, ds.rst.bbox.right, 50)
y_coords = rng.uniform(ds.rst.bbox.bottom, ds.rst.bbox.top, 50)

gdf_int = ds.rst.resample_at_points(
    points=(x_coords, y_coords), resampling="linear", backend="scipy"
)
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

{meth}`ds.rst.reduce_at_points() or Raster.reduce_at_points() <RasterBase.reduce_at_points>`

See {ref}`reducer comparisons and options <transformations-reducers>` to choose the statistic for each window.

```{code-cell} ipython3
gdf_red = ds.rst.resample_at_points(
    (x_coords, y_coords), resampling="median", backend="scipy", window=5
)
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

Choose an {ref}`interpolator <transformations-interpolators>` or {ref}`reducer <transformations-reducers>`
and a backend from the comparisons above.

```{note}
For gridding, GeoUtils introduces nodata values in distances surrounding initial point coordinates, defaulting to a
distance of 1 pixel.
```

{meth}`gdf.pc.grid() or PointCloud.grid() <PointCloudBase.grid>`

```{code-cell} ipython3
# Grid the subsampled points back onto the raster reference
gridded_points = points.pc.grid(ds, resampling="linear", backend="scipy", radius=10)
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
point_marker_size = 50

f, ax = plt.subplots(1, 3, figsize=(14, 5))
ax[0].set_title("Raster, vector\nand points")
_, elevation_cax = ds.rst.plot(ax=ax[0], cmap="terrain", cbar_title="Raster elevation (m)", return_axes=True)
gdf_int.pc.plot(
    ref=ds, ax=ax[0], cmap=point_cmap, add_cbar=False, marker="x", markersize=point_marker_size
)
gdf.vct.plot(ref=ds, ax=ax[0], ec="k", fc="none")
ax[0].legend(
    handles=[
        Line2D([0], [0], color="k", label="Vector contours"),
        Line2D(
            [0], [0], color="darkred", marker="x", markersize=np.sqrt(point_marker_size),
            linestyle="none", label="Points",
        ),
    ],
    loc="lower right",
    fontsize=16,
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
    ref=ds, ax=ax[2], cmap=mask_cmap, vmin=0, vmax=1, cbar_title="Intersects vector",
    marker="x", markersize=point_marker_size, return_axes=True
)
point_cax.set_yticks([0, 1], labels=["False", "True"])
point_cax.yaxis.labelpad = -8
gdf.vct.plot(ref=ds, ax=ax[2], ec="k", fc="none")

_ = ax[1].set_yticklabels([])
_ = ax[2].set_yticklabels([])
for axis in ax:
    axis.xaxis.set_major_locator(MaxNLocator(4))

# Enlarge titles, coordinates and colorbar labels for the three-panel figure
for axis in [*ax, elevation_cax, raster_cax, point_cax]:
    axis.title.set_fontsize(22)
    axis.xaxis.label.set_fontsize(18)
    axis.yaxis.label.set_fontsize(18)
    axis.tick_params(labelsize=16)
    axis.xaxis.get_offset_text().set_fontsize(16)
    axis.yaxis.get_offset_text().set_fontsize(16)
plt.tight_layout()
```

Both outputs can be used as boolean selectors for their reference data.

```{code-cell} ipython3
# Mean of values in the mask
ds.where(raster_mask).mean().item()
```
