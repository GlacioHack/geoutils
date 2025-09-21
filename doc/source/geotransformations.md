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
(geotransformations)=
# Transformations

In GeoUtils, **for all geospatial data objects, georeferenced transformations are exposed through the same functions**
{func}`~geoutils.Raster.reproject`, {func}`~geoutils.Raster.crop` and {func}`~geoutils.Raster.translate`. Additionally,
for convenience and consistency during analysis, most operations can be passed a {class}`~geoutils.Raster` or
{class}`~geoutils.Vector` as a reference to match.
In that case, no other argument is necessary. For more details, see {ref}`core-match-ref`.

```{code-cell} ipython3
:tags: [remove-cell]

# To get a good resolution for displayed figures
from matplotlib import pyplot
pyplot.rcParams['figure.dpi'] = 600
pyplot.rcParams['savefig.dpi'] = 600
pyplot.rcParams['font.size'] = 9
```

## Reproject

{func}`geoutils.Raster.reproject` or {func}`geoutils.Vector.reproject`.

Reprojections **transform geospatial data from one CRS to another**.

For vectors, the transformation of geometry points is exact. However, in the case of rasters, the projected points
do not necessarily fall on a regular grid and require re-gridding by a 2D resampling algorithm, which results in a slight
loss of information (value interpolation, propagation of nodata).

For rasters, it can be useful to use {func}`~geoutils.Raster.reproject` in the same CRS simply for re-gridding,
for instance when downsampling to a new resolution {attr}`~geoutils.Raster.res`.

```{tip}
Due to the loss of information when re-gridding, it is important to **minimize the number of reprojections during the
analysis of rasters** (performing only one, if possible). For the same reason, when comparing vectors and rasters in
different CRSs, it is usually **better to reproject the vector with no loss of information, which is the default
behaviour of GeoUtils in raster–vector–point interfacing**.
```

```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the code for opening example files"
:  code_prompt_hide: "Hide the code for opening example files"

import matplotlib.pyplot as plt
import geoutils as gu
rast = gu.Raster(gu.examples.get_path("everest_landsat_b4"))
rast.set_nodata(0)  # Annoying to have to do this here, should we update it in the example?
rast2 = gu.Raster(gu.examples.get_path("everest_landsat_b4_cropped"))
vect = gu.Vector(gu.examples.get_path("everest_rgi_outlines"))
```

```{code-cell} ipython3
# Reproject vector to CRS of raster by simply passing the raster
vect_reproj = vect.reproject(rast)
# Reproject raster to smaller bounds and different X/Y resolution
rast_reproj = rast.reproject(
    res=(rast.res[0] * 2, rast.res[1] / 2),
    bounds={"left": rast.bounds.left, "bottom": rast.bounds.bottom,
            "right": rast.bounds.left + 10000, "top": rast.bounds.bottom + 10000},
    resampling="cubic")
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

f, ax = plt.subplots(1, 2)
ax[0].set_title("Before reprojection")
rast.plot(ax=ax[0], cmap="gray", add_cbar=False)
vect.plot(rast, ax=ax[0], ec="k", fc="none")
ax[1].set_title("After reprojection")
rast_reproj.plot(ax=ax[1], cmap="gray", add_cbar=False)
vect_reproj.plot(ax=ax[1], ec="k", fc="none")
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
# Reproject vector to CRS of raster by simply passing the raster
rast_reproj2 = rast.reproject(rast2)
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
rast.plot(ax=ax[0], cmap="gray", add_cbar=False)
vect.plot(rast, ax=ax[0], ec="k", fc="none")
ax[1].set_title("Raster 2")
rast2.plot(ax=ax[1], cmap="Reds", add_cbar=False)
vect.plot(rast, ax=ax[1], ec="k", fc="none")
ax[2].set_title("Match-ref\nreprojection")
rast_reproj2.plot(ax=ax[2], cmap="gray", add_cbar=False)
vect_reproj.plot(ax=ax[2], ec="k", fc="none")
_ = ax[1].set_yticklabels([])
_ = ax[2].set_yticklabels([])
plt.tight_layout()
```

## Crop or pad

{func}`geoutils.Raster.crop` or {func}`geoutils.Vector.crop`.

Cropping **modifies the spatial bounds of the geospatial data in a rectangular extent**, by removing or adding data
(in which case it corresponds to padding) without resampling.

For rasters, cropping removes or adds pixels to the sides of the raster grid.

For vectors, cropping removes some geometry features around the bounds, with three options possible:
1. Removing all features **not intersecting** the cropping geometry,
2. Removing all features **not contained** in the cropping geometry,
3. Making all features **exactly clipped** to the cropping geometry (modifies the geometry data).

```{code-cell} ipython3
# Clip the vector to the raster
vect_clipped = vect_reproj.crop(rast, clip=True)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

f, ax = plt.subplots(1, 2)
ax[0].set_title("Before clipping")
rast.plot(ax=ax[0], cmap="gray", add_cbar=False)
vect_reproj.plot(ax=ax[0], ec="k", fc="none")
ax[1].set_title("After clipping")
rast.plot(ax=ax[1], cmap="gray", add_cbar=False)
vect_clipped.plot(ax=ax[1], ec="k", fc="none")
_ = ax[1].set_yticklabels([])
plt.tight_layout()
```

## Translate

{func}`geoutils.Raster.translate` or {func}`geoutils.Vector.translate`.

Translations **modifies the georeferencing of the data by a horizontal offset** without modifying the underlying data,
which is especially useful to align the data due to positioning errors.

```{code-cell} ipython3
# Translate the raster by a certain offset
rast_shift = rast.translate(xoff=1000, yoff=1000)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

f, ax = plt.subplots(1, 2)
ax[0].set_title("Before translation")
rast.plot(ax=ax[0], cmap="gray", add_cbar=False)
vect_clipped.plot(ax=ax[0], ec="k", fc="none")
ax[1].set_title("After translation")
rast_shift.plot(ax=ax[1], cmap="gray", add_cbar=False)
vect_clipped.plot(ax=ax[1], ec="k", fc="none")
_ = ax[1].set_yticklabels([])
plt.tight_layout()
```

:::{admonition} See also
:class: tip

For 3D coregistration tailored to georeferenced elevation data, see [xDEM's coregistration module](https://xdem.readthedocs.io/en/stable/coregistration.html).
:::

## Merge

{func}`geoutils.raster.merge_rasters()`

Merge operations **join multiple geospatial data spatially, possibly with different georeferencing, into a single geospatial
data object**.

For rasters, the merging operation consists in combining all rasters into a single, larger raster. Pixels that overlap
are combined by a reductor function (defaults to the mean). The output georeferenced grid (CRS, transform and shape) can
be set to that of any reference raster (defaults to the extent that contains exactly all rasters).

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for creating multiple raster pieces"
:  code_prompt_hide: "Show the code for creating multiple raster pieces"

# Get 4 cropped bits from initial rasters
rast1 = rast.crop((rast.bounds.left + 1000, rast.bounds.bottom + 1000,
                   rast.bounds.left + 3000, rast.bounds.bottom + 3000))
rast2 = rast.crop((rast.bounds.left + 3000, rast.bounds.bottom + 1000,
                   rast.bounds.left + 5000, rast.bounds.bottom + 3000))
rast3 = rast.crop((rast.bounds.left + 1000, rast.bounds.bottom + 3000,
                   rast.bounds.left + 3000, rast.bounds.bottom + 5000))
rast4 = rast.crop((rast.bounds.left + 3000, rast.bounds.bottom + 3000,
                   rast.bounds.left + 5000, rast.bounds.bottom + 5000))
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
