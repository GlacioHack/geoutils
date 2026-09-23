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
(quick-start)=

# Quick start

A short code example using several end-user functionalities of GeoUtils. For a more detailed example of all features,
have a look at the {ref}`feature-overview` page! Or, to find an example about
a specific functionality, jump to {ref}`quick-gallery` right below.

```{code-cell} ipython3
:tags: [remove-cell]

# To get a good resolution for displayed figures
from matplotlib import pyplot
pyplot.rcParams['figure.dpi'] = 600
pyplot.rcParams['savefig.dpi'] = 600
pyplot.rcParams['font.size'] = 9
```

## Download data examples

Example data from GeoUtils can be automatically downloaded with the function `geoutils.examples.get_path()` as used below.

See the {ref}`data` page to learn more about all our example data and different download options.

## Short example

```{note}
:class: margin

Most functions shown here normally require many lines of code using several independent packages with inconsistent
geospatial syntax and volatile passing of metadata that can lead to errors!

In GeoUtils, **these higher-level operations are tested to ensure robustness and consistency**. 🙂
```

The package functionalities can be called either from the {class}`rst <geoutils.RasterAccessor>`,
{class}`vct <geoutils.VectorAccessor>` and {class}`pc <geoutils.PointCloudAccessor>` accessors or from the
{class}`~geoutils.Raster`, {class}`~geoutils.Vector` and {class}`~geoutils.PointCloud` classes.
Below, in a few lines, we load raster, vector and point data, crop them to a common extent, re-assign raster values
around a buffer of the vector, sample values, calculate statistics, and finally plot and save the result!

<!--- An empty margin to add some vertical padding --->
```{margin}
&nbsp;
```


```{note}
:class: margin

**A `UserWarning` appears:** No nodata value was defined in the GeoTIFF file, so GeoUtils automatically defined
one compatible with the data type derived during operations.
```

```{code-cell} ipython3
import geoutils as gu
import numpy as np

# Example files: paths to raster, vector and point cloud files
filename_rast = gu.examples.get_path("everest_landsat_b4")
filename_vect = gu.examples.get_path("everest_rgi_outlines")
filename_pc = gu.examples.get_path("coromandel_lidar")
```

::::{tab-set}
:::{tab-item} With accessors

```python
# Open native Xarray and GeoPandas objects; raster and point data stay lazy and chunked
ds = gu.open_raster(filename_rast, chunks={"x": 512, "y": 512})
gdf = gu.open_vector(filename_vect)
points = gu.open_pointcloud(filename_pc, data_column="Z", chunks=5_000)

# Crop raster to vector's extent by simply passing vector as "match-reference"
ds = ds.rst.crop(gdf)

# Buffer the vector by 500 meters no matter its current projection system
gdf_buff = gdf.vct.buffer_metric(500)

# Create mask of vector on the same grid/CRS as raster using it as "match-reference"
mask_buff = gdf_buff.vct.create_mask(ds)

# Re-assign values of pixels in the mask while performing a sum
calc_ds = ds.where(~mask_buff, ds + 50)
calc_ds = np.log(calc_ds / 2) + 3.5

# Sample the raster and calculate raster and point statistics with chunked execution
sample = calc_ds.rst.subsample(subsample=500)
raster_stats = calc_ds.rst.stats(["mean", "nmad"])
point_stats = points.pc.stats(["mean", "nmad"])
print(raster_stats, point_stats, sample.compute().head())

# Plot a bounded raster view without loading all pixels, using raster as projection-reference for vector
calc_ds.rst.plot(max_pixels=500_000, cmap="Spectral", cbar_title="My calculation")
gdf_buff.vct.plot(calc_ds, fc="none", ec="k", lw=0.5)

# Compute and save the lazy raster chunk-by-chunk
calc_ds.rst.to_file("mycalc_accessor.tif", compute=True)
```

:::
:::{tab-item} Without accessors

```python
from geoutils.multiproc import MultiprocConfig

# Open GeoUtils objects; raster and point data remain unloaded until required
rast = gu.Raster(filename_rast)
vect = gu.Vector(filename_vect)
point_cloud = gu.PointCloud(filename_pc, data_column="Z")

# Crop raster to vector's extent by simply passing vector as "match-reference"
rast = rast.crop(vect)

# Buffer the vector and create a mask on the raster grid
vect_buff = vect.buffer_metric(500)
mask_buff = vect_buff.create_mask(rast)

# Sample and calculate statistics in chunks with Multiprocessing
raster_config = MultiprocConfig(chunks=512, outfile="quick_sample.gpkg")
point_config = MultiprocConfig(chunks=5_000)
sample = rast.subsample(subsample=500, mp_config=raster_config)
raster_stats = rast.stats(["mean", "nmad"], mp_config=raster_config)
point_stats = point_cloud.stats(["mean", "nmad"], mp_config=point_config)
print(raster_stats, point_stats, sample)

# Re-assign values of pixels in the mask while performing a sum
rast[mask_buff] = rast[mask_buff] + 50
calc_rast = np.log(rast / 2) + 3.5

# Plot a bounded raster view without loading all pixels, using raster as projection-reference for vector
calc_rast.plot(max_pixels=500_000, cmap="Spectral", cbar_title="My calculation")
vect_buff.plot(calc_rast, fc="none", ec="k", lw=0.5)

# Save to file
calc_rast.to_file("mycalc_object.tif")
```

:::
::::

```{code-cell} ipython3
:tags: [remove-cell]
import os
for filename in ("mycalc_accessor.tif", "mycalc_object.tif", "quick_sample.gpkg"):
    if os.path.exists(filename):
        os.remove(filename)
```

(quick-gallery)=
## More examples

To dive into more illustrated code, explore our gallery of examples that is composed of:
- An {ref}`examples-io` section on opening, saving, loading, importing and exporting,
- An {ref}`examples-handling` section on geotransformations (crop, reproject) and raster-vector interfacing,
- An {ref}`examples-analysis` section on analysis tools and raster numerics.

See also the full concatenated list of examples below.

```{eval-rst}
.. minigallery:: geoutils.RasterAccessor geoutils.VectorAccessor geoutils.PointCloudAccessor
    :add-heading: Examples using rasters, vectors and point clouds
```
