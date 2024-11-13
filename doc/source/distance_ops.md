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
(distance-ops)=
# Distance operations

Computing distance between sets of geospatial data or manipulating their shape based on distance is often important
for later analysis. To facilitate this type of operations, GeoUtils implements distance-specific functionalities
for both vectors and rasters.

```{code-cell} ipython3
:tags: [remove-cell]

# To get a good resolution for displayed figures
from matplotlib import pyplot
pyplot.rcParams['figure.dpi'] = 600
pyplot.rcParams['savefig.dpi'] = 600
pyplot.rcParams['font.size'] = 9
```

```{tip}
It is often important to compute distances in a metric CRS. For this, reproject (with
{func}`~geoutils.Raster.reproject`) to a local metric CRS (that can be estimated with {func}`~geoutils.Raster.get_metric_crs`).
```

## Proximity

Proximity corresponds to **the distance to the closest target geospatial data**, computed on each pixel of a raster's grid.
The target geospatial data can be either a vector or a raster.

{func}`geoutils.Raster.proximity` and {func}`geoutils.Vector.proximity`

```{code-cell} ipython3
:tags: [hide-cell]
:mystnb:
:  code_prompt_show: "Show the code for opening example files"
:  code_prompt_hide: "Hide the code for opening example files"

import matplotlib.pyplot as plt
import geoutils as gu
import numpy as np

rast = gu.Raster(gu.examples.get_path("everest_landsat_b4"))
rast.set_nodata(0)  # Annoying to have to do this here, should we update it in the example?
vect = gu.Vector(gu.examples.get_path("everest_rgi_outlines"))
```

```{code-cell} ipython3
# Compute proximity to vector outlines
proximity = vect.proximity(rast)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

f, ax = plt.subplots(1, 2)
ax[0].set_title("Raster and vector")
rast.plot(ax=ax[0], cmap="gray", add_cbar=False)
vect.plot(ref_crs=rast, ax=ax[0], ec="k", fc="none")
ax[1].set_title("Proximity")
proximity.plot(ax=ax[1], cmap="viridis", cbar_title="Distance to outlines (m)")
_ = ax[1].set_yticklabels([])
plt.tight_layout()
```

## Buffering without overlap

Buffering consists in **expanding or collapsing vector geometries equally in all directions**. However, this can often lead to overlap
between shapes, which is sometimes undesirable. Using Voronoi polygons, we provide a buffering method without overlap.

{func}`geoutils.Vector.buffer_without_overlap`

```{code-cell} ipython3
# Compute buffer without overlap from vector exterior
vect_buff_nolap = vect.buffer_without_overlap(buffer_size=500)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

# Plot with color to see that the attributes are retained for every feature
vect.plot(ax="new", ec="k", column="Area", alpha=0.5, add_cbar=False)
vect_buff_nolap.plot(column="Area", cbar_title="Buffer around initial features\ncolored by glacier area (km)")
```
