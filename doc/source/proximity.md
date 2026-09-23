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
(proximity)=
# Proximity

GeoUtils integrates proximity features to compute **distances and neighborhoods within or between** sets of geospatial data,
with **scalable execution** on large datasets.

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
{meth}`reproject() <RasterBase.reproject>`) to a local metric CRS (that can be estimated
with {meth}`get_metric_crs() <RasterBase.get_metric_crs>`).
```

## Proximity

{meth}`ds.rst.proximity() or Raster.proximity() <RasterBase.proximity>`<br>
{meth}`gdf.vct.proximity() or Vector.proximity() <VectorBase.proximity>`

Proximity corresponds to **the distance to the closest target geospatial data**, computed on each pixel of a raster's grid.
The target geospatial data can be either a vector or a raster.

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
# Compute proximity to vector outlines
proximity = gdf.vct.proximity(ds.rst)
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
ax[1].set_title("Proximity to vector")
proximity.rst.plot(ax=ax[1], cmap="viridis", cbar_title="Distance to outlines (m)")
_ = ax[1].set_yticklabels([])
plt.tight_layout()
```

## Buffering without overlap

{meth}`gdf.vct.buffer_without_overlap() or Vector.buffer_without_overlap() <VectorBase.buffer_without_overlap>`

Buffering without overlap consists in **expanding or collapsing vector geometries equally in all directions while preventing overlap**.

```{code-cell} ipython3
# Compute buffer without overlap from vector exterior
gdf_buff_nolap = gdf.vct.buffer_without_overlap(buffer_size=500)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show the code for plotting the figure"
:  code_prompt_hide: "Hide the code for plotting the figure"

# Plot with color to see that the attributes are retained for every feature
gdf.vct.plot(ax="new", ec="k", column="Area", alpha=0.5, add_cbar=False)
gdf_buff_nolap.vct.plot(column="Area", cbar_title="Buffer around initial features\ncolored by glacier area (km)")
```
