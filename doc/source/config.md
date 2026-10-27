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
(config)=
# Configuration

You can **configure the default behaviour of GeoUtils at the package level** for operations that depend on user preference
(such as resampling method, default data structure returned, or behaviour around pixel interpretation).

## Changing configuration during a session

Using a global configuration setting **ensures operations will always be performed consistently**, even when used
under-the-hood by higher-level methods (such as {ref}`Cosampling <sampling-cosample>` used by {ref}`Statistics <stats>` within GeoUtils, or [Coregistration](https://xdem.readthedocs.io/en/stable/coregistration.html) in other packages like [xDEM](https://xdem.readthedocs.io/en/stable/)),
without having to rely on multiple keyword arguments to pass to subfunctions.

For example, choose the **default resampling method** for each type of input data (raster or point).
Both settings default to `"linear"` and can be overridden by passing `resampling=` to an operation.

For raster input, `raster_resampling_method` applies to reprojection and resampling at points:

```{code-cell}
import geoutils as gu

gu.config["raster_resampling_method"] = "cubic"
```

For point cloud input, `point_resampling_method` applies to gridding and resampling at other points:

```{code-cell}
gu.config["point_resampling_method"] = "nearest"
```

Or choose the return type for all newly created point clouds (e.g. in
{meth}`to_pointcloud() <RasterBase.to_pointcloud>` and
{meth}`interp_at_points() <RasterBase.interp_at_points>`):

```{code-cell}
gu.config["pointcloud_return_type"] = "geodataframe"
```

The default is `"dataarray"`. Use `"dataset"` for an Xarray Dataset, or `"geodataframe"` for GeoPandas.

For newly created rasters, choose between `"dataarray"` (the default) and `"dataset"`:

```{code-cell}
gu.config["raster_return_type"] = "dataset"
```

This setting applies to {func}`~geoutils.open_raster`, {func}`~geoutils.raster.from_array`, and rasters created
from point clouds or vectors through accessors. Pass `as_object_type=` to either function to override the setting.
Operations on existing raster DataArrays and Datasets preserve their container type.

The **densification of projected bounds and footprints** is controlled by `densify_points` (default: 5000).
Configure it for rasters, vectors and point clouds:

```{code-cell}
gu.config["densify_points"] = 100
```

Passing `densify_points=` to `get_bbox_projected()`, `get_footprint_projected()` or
`Vector.from_bounds_projected()` overrides this setting. Raster bounding boxes limit the count to the larger
raster dimension in pixels.

Advanced users can configure warnings when comparing rasters with different
{ref}`pixel interpretations <pixel-interpretation>`. These warnings are disabled by default:

```{code-cell}
# Default behaviour for pixel interpretation
gu.config["warn_area_or_point"] = False
```

## Default configuration file

Below is the **default configuration file**.
Note that changes to `gu.config` apply only to the current Python session, they do not modify this file!

```{literalinclude} ../../geoutils/config.ini
:class: full-width
```
