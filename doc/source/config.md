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

## Default configuration file

Below is the full default configuration file, which is updated by changes in configuration during a session.

```{literalinclude} ../../geoutils/config.ini
:class: full-width
```
