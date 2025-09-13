---
file_format: mystnb
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: xdem-env
  language: python
  name: xdem
---
(point-cloud)=

# The georeferenced point cloud ({class}`~geoutils.PointCloud`)

A point cloud represents 2D point geometries of georeferenced coordinates associated with a main 1D data array, and optionally auxiliary data.

Although a subtype of {class}`~geoutils.Vector`, point clouds have a very different nature than other vectors and are
ubiquitous in geospatial analysis, requiring their own object type.
For **numerical operations**, a point cloud manipulation is facilitated by its own arithmetic to manipulate its main data array, as well as a
specific interface with rasters (interpolation or reduction to same coordinates, or point gridding), or other point
clouds (pairwise distance matching).
For **geometric operations**, point clouds interface in specific ways with other vector-types (zonal statistics, geometric masking).

GeoUtils aims to support these features, as well the reading and writing of point clouds both from vector-type files (e.g., ESRI shapefile, geopackage,
geoparquet) usually used for **sparse point clouds**, and from point-cloud-type files (e.g., LAS, LAZ, COPC) usually
used for **dense point clouds**.

TODO: ADD GUIDE PAGE ON POINT CLOUD TYPES AND RASTER COMPARISON?

Below, a summary of the {class}`~geoutils.PointCloud` object and its methods.

(pc-obj-def)=

## Object definition and attributes

A {class}`~geoutils.PointCloud` is a {class}`~geoutils.Vector` is a vector of 2D point geometries associated to
numeric values from a main {attr}`~geoutils.PointCloud.data` column, and can also contain auxiliary data columns.

It inherits the main {class}`~geoutils.Vector` attribute {attr}`~geoutils.Vector.ds` containing the geodataframe, and adds **another
main attribute** {attr}`~geoutils.PointCloud.data_column` that identifies the name of the main data associated to the
point geometries.

New derivatives attributes and methods specific to point cloud are detailed further below.

Generic vector attributes and methods are inherited through the {class}`~geoutils.Vector` object, such as
{attr}`~geoutils.Vector.bounds`, {attr}`~geoutils.Vector.crs`, {func}`~xdem.Vector.reproject` and {func}`~xdem.Vector.crop`.

```{tip}
The complete list of {class}`~geoutils.Vector` attributes and methods can be found in [the Vector section of the API](https://geoutils.readthedocs.io/en/stable/api.html#vector).
```

## Open and save

A {class}`~geoutils.PointCloud` is opened by instantiating {class}`str`, a {class}`pathlib.Path`, a {class}`geopandas.GeoDataFrame`,
a {class}`geopandas.GeoSeries` or a {class}`shapely.Geometry`, as for a {class}`~geoutils.Raster`.

```{code-cell} ipython3
:tags: [remove-cell]

# To get a good resolution for displayed figures
from matplotlib import pyplot
pyplot.rcParams['figure.dpi'] = 400
pyplot.rcParams['savefig.dpi'] = 400
```

```{code-cell} ipython3
import xdem

# Instantiate a DEM from a filename on disk
filename_dem = xdem.examples.get_path("longyearbyen_ref_dem")
dem = xdem.DEM(filename_dem)
dem
```

## Create from arrays or tuples

## Plotting

## Gridding

## Arithmetic

## Array interface

## Statistics
