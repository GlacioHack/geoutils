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

You can configure the default behaviour of GeoUtils at the package level for operations that depend on user preference
(such as resampling method, or pixel interpretation).

## Changing configuration during a session

Using a global configuration setting ensures operations will always be performed consistently, even when used
under-the-hood by higher-level methods (such as [Coregistration](https://xdem.readthedocs.io/en/stable/coregistration.html)),
without having to rely on multiple keyword arguments to pass to subfunctions.

```{code-cell}
import geoutils as gu
# Changing default behaviour for pixel interpretation for this session
gu.config["shift_area_or_point"] = False
```

Or choose the return type for all newly created point clouds (e.g. in
{meth}`to_pointcloud() <RasterBase.to_pointcloud>` and
{meth}`interp_at_points() <RasterBase.interp_at_points>`):

```{code-cell}
gu.config["pointcloud_return_type"] = "geodataframe"
```

The default is `"dataarray"`. Use `"dataset"` for an Xarray Dataset, or `"geodataframe"` for GeoPandas.
Override it for one conversion with `ds.rst.to_pointcloud(pointcloud_return_type="dataarray")`.

## Default configuration file

Below is the default configuration file. Changes to `gu.config` apply only to the current Python session,
they do not modify this file.

```{literalinclude} ../../geoutils/config.ini
:class: full-width
```
