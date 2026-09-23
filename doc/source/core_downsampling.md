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
# Downsampling for opening and plotting

```{code-cell} ipython3
:tags: [remove-cell]

# Match the figure resolution and text size used by the other feature pages
from matplotlib import pyplot as plt
plt.rcParams["figure.dpi"] = 600
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["font.size"] = 9
```

Rasters and point clouds can both contain much more data than an analysis or a figure needs. GeoUtils provides two
levels of reduction with consistent roles across these objects:

- `downsample` reduces the data represented by an object when it is opened;
- `max_pixels` or `max_points` reduces only the temporary data passed to Matplotlib by `plot()`.

## Reduce data when opening

Pass `downsample` when later operations should use a smaller dataset:

```{code-cell} ipython3
import geoutils as gu

raster_path = gu.examples.get_path("exploradores_aster_dem")
point_path = gu.examples.get_path("coromandel_lidar")

raster = gu.open_raster(raster_path, downsample=4)
points = gu.open_pointcloud(point_path, data_column="Z", downsample=4)
```

The factor has a meaning appropriate to each data type:

```{list-table}
:header-rows: 1
:widths: 25 45 30

* - Object
  - Effect of `downsample=4`
  - Approximate size
* - {func}`~geoutils.open_raster`
  - Multiply the source pixel size by 4 in both directions
  - One sixteenth of the source pixels
* - {func}`~geoutils.open_pointcloud`
  - Select a deterministic random sample of the complete point rows
  - `ceil(point_count / 4)` points
```

For rasters, rows or columns that do not fill a complete interval are omitted. For point clouds, the selected rows
keep their geometries, main data values and auxiliary columns. A factor of 1, the default, keeps the native raster grid
or every point. Opening with `downsample` does not alter the source file.

For point cloud files opened eagerly, the sample is selected after the rows are read. It reduces the resulting object
and the cost of later operations, but does not reduce the peak memory needed by the initial file read. Pass `chunks`
to {func}`~geoutils.open_pointcloud` when that initial read must also stay partitioned and lazy.

The same options are available when opening the corresponding {class}`~geoutils.Raster` and
{class}`~geoutils.PointCloud` objects.

## Reduce data only for plotting

The default `plot()` behavior limits the amount of data sent to Matplotlib while leaving the source object unchanged:

```{code-cell} ipython3
raster.rst.plot(max_pixels="auto")
points.pc.plot(max_points="auto", ax="new")
```

For a raster, `max_pixels="auto"` limits the display grid to the Matplotlib axes width and height in display pixels,
as determined by the figure size and DPI. An integer sets a maximum total pixel count. The grid keeps the raster
aspect ratio and is produced through {meth}`~RasterBase.reproject` with the selected `resampling` method.

For a point cloud, `max_points="auto"` limits the sample to the smaller of the Matplotlib axes pixel area and one
million points. An integer sets the maximum sample size, and `random_state` controls the random point selection. The
default seed of 0 makes repeated plots select the same rows.

Pass `None` to either argument to use the native raster grid or every point. With Dask-backed inputs, preparation
remains lazy until `plot()` requests the reduced display data.

Use opening `downsample` when all later operations should work with less data. Use the plotting limit when only the
figure needs fewer pixels or points.

## Raster file overviews

Some raster formats can store reduced-resolution copies called
[**overviews**](https://rasterio.readthedocs.io/en/stable/topics/overviews.html). To open one of these copies through
Rioxarray, pass `overview_level` to {func}`~geoutils.open_raster`:

```{code-cell} ipython3
:tags: [remove-cell]

import shutil

import rasterio as rio
from rasterio.enums import Resampling

overview_path = "overview_example.tif"
shutil.copyfile(raster_path, overview_path)
with rio.open(overview_path, "r+") as dataset:
    dataset.build_overviews([2], Resampling.nearest)
```

```{code-cell} ipython3
raster_overview = gu.open_raster(overview_path, overview_level=0)
```

Level 0 selects the first stored overview, level 1 the second, and so on. The returned raster has the shape and
resolution of that stored overview. Rioxarray shows the same argument in its
[Cloud Optimized GeoTIFF example](https://corteva.github.io/rioxarray/stable/examples/COG.html).

Overviews are also used automatically with `downsample`. GeoUtils selects the closest suitable stored overview and
resamples it to the grid defined by the requested factor. It reads the original raster when no suitable overview
exists. This behavior is the same for {class}`~geoutils.Raster` and {func}`~geoutils.open_raster`.

`downsample` and `overview_level` cannot be combined because one selects an overview automatically and the other does
so explicitly:

- `downsample` defines the final shape, coordinates and resolution. A stored overview is used as the source for that
  grid when possible.
- `overview_level` returns the selected overview's own stored grid and is available through {func}`~geoutils.open_raster`.
- `max_pixels` creates a temporary grid for {meth}`~RasterBase.plot` and does not explicitly select a stored
  overview.

```{code-cell} ipython3
:tags: [remove-cell]

from pathlib import Path

raster_overview.close()
Path(overview_path).unlink()
```
