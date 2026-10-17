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
(core-downsampling)=
(scalability-downsampling)=
# Downsampling and overviews

GeoUtils supports **downsampling** and **overview-reading** to reduce large data for analysis or plotting.

Downsampling should be seen as an **alternative** to processing the full dataset when a lower level of detail is sufficient,
for example when finer detail would anyway not be visible given a figure's resolution, or when a reduced sample is still large
enough to estimate statistics of interest.

Downsampling can be performed **on-the-fly** during operations such as opening, plotting or specific functionalities.
Formats such as COG and COPC can also store **overviews or levels of detail**. GeoUtils reads these automatically
when downsampling, or lets you select them explicitly.

Below, we describe these features:

- {ref}`scalability-downsampling-opening` selects a **coarser grid or point sample**,
- {ref}`scalability-downsampling-plotting` automatically reduces the data displayed in **a figure**,
- {ref}`scalability-downsampling-operations` selects a random **subsample** specific to a functionality,
- {ref}`scalability-downsampling-overviews` provide **coarser raster copies** for faster reading,
- {ref}`scalability-downsampling-copc` reads **coarser point cloud detail** from a spatial hierarchy.

```{seealso}
For more details on file formats and their use in the cloud, see {ref}`scalability-formats`.
```

(scalability-downsampling-opening)=
## Downsampling when opening

The {func}`~geoutils.open_raster` and {func}`~geoutils.open_pointcloud` functions can be passed a `downsample` argument, which defines a
**downsampling factor**, with a slightly different meaning for each data type.

For **rasters**, `downsample=4` samples one out of every four pixels in each dimension, dividing the total pixel
count by about 16. A floating factor also works: `downsample=2.5` turns 30 m pixels into 75 m pixels in each
dimension, with nearest-neighbor resampling selecting a value for each output pixel from the source grid or a suitable
{ref}`stored overview <scalability-downsampling-overviews>`.

```{note}
With **Dask**, passing `downsample` and `chunks` together when opening also allows later operations to process the
reduced dataset in chunks.
```

The examples use the 30 m Exploradores DEM and the Coromandel LiDAR survey.

```{code-cell} ipython3
:tags: [remove-cell]

import numpy as np
import rasterio as rio
from matplotlib import pyplot as plt

# Match the figure style of other feature pages
plt.rcParams["figure.dpi"] = 600
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["font.size"] = 9
```

```{code-cell} ipython3
import geoutils as gu

raster_path = gu.examples.get_path("exploradores_aster_dem")
ds = gu.open_raster(raster_path, chunks={"x": 128, "y": 128})
ds_coarse = gu.open_raster(raster_path, downsample=4, chunks={"x": 32, "y": 32})
print("Native raster pixels:", ds.size)
print("Downsampled raster pixels:", ds_coarse.size)
print("Sample stays lazy:", ds_coarse.chunks is not None)
```

For **point clouds**, using `downsample` selects a random sample of size the total number of points divided by the downsampling factor.
For example, `downsample=2.5` selects about 40% of the points.
For **COPC**, GeoUtils first selects {ref}`levels of detail <scalability-downsampling-copc>` from coarse to fine
until enough points are available, then samples those points to the requested count.

```{code-cell} ipython3
point_path = gu.examples.get_path("coromandel_lidar")
ds_lidar = gu.open_pointcloud(point_path, data_name="Z", columns=["classification"], chunks=100_000)
ds_sample = gu.open_pointcloud(point_path, data_name="Z", columns=["classification"], downsample=4, chunks=100_000)
print("Native points:", ds_lidar.pc.point_count)
print("Sampled points:", ds_sample.pc.point_count)
print("Sample stays lazy:", ds_sample.chunks is not None)
```

For rasters, dimensions are rounded down to complete sampling intervals, omitting an incomplete interval
at the right or bottom edge. Downsampling leaves the source file unchanged, and is also available
when opening {class}`~geoutils.Raster` and {class}`~geoutils.PointCloud` objects.

```{code-cell} ipython3
:tags: [remove-cell]

# Check the reduced grid and that eager and lazy point samples select the same complete rows
assert ds_coarse.rst.shape == (154, 134) and ds_coarse.rst.res == (120, 120)
assert not ds._in_memory and not ds_coarse._in_memory
assert ds_coarse.rst.crs == ds.rst.crs
assert ds_coarse.rst.bbox.left == ds.rst.bbox.left and ds_coarse.rst.bbox.top == ds.rst.bbox.top
assert ds_lidar.pc.point_count == 584348 and ds_sample.pc.point_count == 146087
assert not ds_lidar._in_memory and not ds_sample._in_memory
sample_eager = gu.open_pointcloud(point_path, data_name="Z", columns=["classification"], downsample=4)
sample_computed = ds_sample.compute()
np.testing.assert_array_equal(sample_computed, sample_eager)
np.testing.assert_array_equal(sample_computed.x, sample_eager.x)
np.testing.assert_array_equal(sample_computed.classification, sample_eager.classification)
```

(scalability-downsampling-plotting)=
## Downsampling when plotting

The {meth}`~RasterBase.plot` function **automatically downsamples rasters and point clouds to the figure's size
and resolution**, omitting detail too fine to see in the plot.

By default, `"auto"` determines the limit from the axes size and figure DPI, with a maximum of 100,000 points.
The maximum number of pixels or points used can also be controlled manually with the `max_pixels` and `max_points` arguments.
Passing `None` displays the original grid or every point.

Below, we plot the raster and point cloud with at most 50,000 pixels or points:

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(7, 3))
ds.rst.plot(ax=axes[0], max_pixels=50_000, cmap="terrain")
ds_lidar.pc.plot(ax=axes[1], max_points=50_000, cmap="terrain", s=4)
```

```{code-cell} ipython3
print("Source raster pixels:", ds.size)
print("Displayed raster pixels:", axes[0].images[0].get_array().size)
print("Source points:", ds_lidar.pc.point_count)
print("Displayed points:", len(axes[1].collections[0].get_offsets()))
```

For a preview of a specific area at **original resolution** on a large dataset, simply select a window with {meth}`~RasterBase.crop` (which will defer loading to this area only),
then call {meth}`~RasterBase.plot`!

(scalability-downsampling-operations)=
## Downsampling within operations

Some forms of downsampling are also **directly supported in certain operations**.

Most of these operations are statistical or conversions, such as {meth}`~RasterBase.stats`,
{meth}`~RasterBase.variogram`, {meth}`~RasterBase.estimate_error_structure`, or {meth}`~RasterBase.to_pointcloud`.
Under-the-hood, these operations rely on methods from **the {ref}`sampling` module**.

For example, we can estimate the **median and NMAD** from at most 5,000 valid elevations:

```{code-cell} ipython3
ds.rst.stats(["median", "nmad"], subsample=5_000, random_state=42)
```

We can also estimate spatial variability from 2,000 pairs:

```{code-cell} ipython3
variogram = ds.rst.variogram(n_pairs=2_000, n_lags=10, random_state=42)
variogram.to_dataframe().head()
```

(scalability-downsampling-overviews)=
## Raster overviews

GeoUtils **reads suitable raster overviews** in GeoTIFFs/COGs through [Rasterio](https://rasterio.readthedocs.io/en/stable/) when specifying an overview level,
or when opening with `downsample`,.

[Overviews](https://rasterio.readthedocs.io/en/stable/topics/overviews.html) are coarser copies stored inside or beside a raster file, avoid reading and
resampling the full-resolution data.

**To read a specific stored overview level** , use `overview_level`, starting at 0.
Available overview levels and their resampling can be inspected from the raster attributes without loading the file.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show overview preparation"
:  code_prompt_hide: "Hide overview preparation"

from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

from rasterio.enums import Resampling

# Add overviews to a separate copy, preserving the original file for the examples above
files = TemporaryDirectory()
overview_path = Path(files.name) / "elevation_overviews.tif"
copyfile(raster_path, overview_path)
with rio.open(overview_path, "r+") as source:
    source.build_overviews([2, 4, 8], Resampling.average)
    source.update_tags(ns="rio_overview", resampling="average")
```

```{code-cell} ipython3
# Level 1 selects the second overview, reduced by a factor of 4
ds_overview = gu.open_raster(overview_path, overview_level=1, chunks={"x": 32, "y": 32})
print("Stored factors:", ds_overview.attrs["overview_factors"])
print("Overview resampling:", ds_overview.attrs["overview_resampling"])
```

**When opening with `downsample`**, GeoUtils automatically chooses the coarsest overview whose reduction factor is at most `1.2 × downsample`.
The selected overview is then resampled exactly to the downsampling factor. Downsampling cannot be used together with `overview_level`.

```{code-cell} ipython3
# A requested factor of 3 instead defines its own 90 m grid
ds_factor3 = gu.open_raster(overview_path, downsample=3, chunks={"x": 32, "y": 32})
print("Overview factor selected for downsample=3:", ds_factor3.attrs["overview_selected_factor"])
print("Stored overview pixels:", ds_overview.size)
print("downsample=3 pixels:", ds_factor3.size)
```

During downsampling, GeoUtils **stores which overview was selected** for future inspection.

See {ref}`scalability-formats` for reading remote raster files in the cloud, including with overviews.

```{code-cell} ipython3
:tags: [remove-cell]

# Compare the opened overview with the stored copy, including missing values
with rio.open(overview_path, overview_level=1) as source:
    stored_overview = source.read(1, masked=True).astype("float32").filled(np.nan)
    assert ds_overview.rst.shape == source.shape
np.testing.assert_array_equal(ds_overview.compute(), stored_overview)
assert ds_factor3.rst.shape == (206, 179) and ds_factor3.rst.res == (90, 90)
assert not ds_overview._in_memory and not ds_factor3._in_memory
for array in (
    ds, ds_coarse, ds_lidar, ds_sample, sample_eager, sample_computed, ds_overview, ds_factor3,
):
    array.close()
files.cleanup()
```

(scalability-downsampling-copc)=
## Point cloud levels of detail

GeoUtils **reads suitable levels of detail** in COPCs through [LasPy](https://laspy.readthedocs.io/en/latest/) when specifying a level of detail,
or when opening with `downsample`.

[Levels of detail](https://copc.io/) are coarser point samples stored in a COPC's octree, a hierarchy of progressively smaller spatial cells,
avoiding reading every point. They select original points, preserving their coordinates, values and attributes.

**To read a specific stored level of detail**, use `level`, starting at 0.
An integer selects only that level, a range combines several levels into one point cloud, and all levels is equivalent to the full point cloud.
Available levels of detail can be inspected from the point cloud attributes without loading the file.

```{code-cell} ipython3
:tags: [remove-cell]

copc_path = Path("_static/survey.copc.laz")
```

```{code-cell} ipython3
ds_coarse = gu.open_pointcloud(copc_path, level=0, chunks=10_000)
ds_detail = gu.open_pointcloud(copc_path, level=range(2), chunks=10_000)
print("Available levels:", ds_coarse.attrs["copc_levels"])
print("Level 0 points:", ds_coarse.pc.point_count)
print("Levels 0 and 1 points:", ds_detail.pc.point_count)
```

**When opening with `downsample`**, GeoUtils automatically selects levels from coarse to fine until enough points are available for the requested count.
The selected points are then sampled exactly to the downsampling factor. Downsampling cannot be used together with `level`.

```{code-cell} ipython3
ds_copc_sample = gu.open_pointcloud(copc_path, downsample=5, chunks=10_000)
print("Levels selected for downsample=5:", ds_copc_sample.attrs["copc_selected_levels"])
print("Sampled points:", ds_copc_sample.pc.point_count)
```

As for raster overviews, during downsampling of point clouds, GeoUtils **stores which levels were selected**.

See {ref}`scalability-formats` for reading remote point cloud files in the cloud, including with levels of detail.

```{code-cell} ipython3
:tags: [remove-cell]

# Check that selecting more levels adds points and that the selected arrays stay lazy
assert not ds_coarse.pc.is_loaded and not ds_detail.pc.is_loaded
assert 0 < ds_coarse.pc.point_count < ds_detail.pc.point_count == 100_003
assert ds_coarse.compute().size == ds_coarse.pc.point_count
assert ds_detail.compute().size == ds_detail.pc.point_count

# Check automatic downsampling uses only level 0 and samples it to the requested count
assert not ds_copc_sample.pc.is_loaded
assert ds_copc_sample.attrs["copc_levels"] == [0, 1]
assert ds_copc_sample.attrs["copc_selected_levels"] == [0]
assert ds_copc_sample.pc.point_count == int(np.ceil(ds_detail.pc.point_count / 5))
assert ds_copc_sample.compute().size == ds_copc_sample.pc.point_count
ds_coarse.close()
ds_detail.close()
ds_copc_sample.close()
```
