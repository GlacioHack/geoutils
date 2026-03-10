(cheatsheet-osgeo)=
# Cheatsheet: From GDAL/OGR

This page helps users familiar with **GDAL/OGR** migrate their operations to the GeoUtils API.

Regarding function names, GeoUtils exposes **an API almost entirely consistent with the recently overhauled GDAL CLI**. 

Note that GeoUtils is **object-oriented** (methods run on {class}`~geoutils.Raster`, {class}`~geoutils.Vector`, {class}`~geoutils.PointCloud`, or on {class}`~xarray.DataArray` and {class}`~geopandas.GeoDataFrame` through {class}`rst <geoutils.RasterAccessor>`, `vct` and `pc` accessors), while GDAL/OGR utilities are typically **file-oriented** (read from disk, write to disk).

We also provide a conversion table for operations specific to DEMs (e.g. slope, aspect, roughness indexes) that are supported through our sister-package [xDEM](https://xdem.readthedocs.io/en/stable/).

## GDAL/OGR utilities

```{list-table} GDAL/OGR ⟶ GeoUtils
:header-rows: 1
:widths: 3 4 4 2
:align: left
:class: tight-table

* - Operation
  - Old GDAL/OGR
  - New GDAL CLI
  - GeoUtils

* - <span class="gu-table-section">Metadata</span>
  -
  -
  -

* - Footprint
  - `gdalinfo`/`ogrinfo`
  - `gdal raster/vector footprint`
  - {attr}`~geoutils.Raster.footprint`

* - Bounding box
  - `gdalinfo`/`ogrinfo`
  - `gdal raster/vector bbox`
  - {attr}`~geoutils.Raster.bounds`

* - Info summary
  - `gdalinfo`/`ogrinfo`
  - `gdal raster/vector info`
  - {attr}`~geoutils.Raster.info`

* - <span class="gu-table-section">Raster ⟶ Raster</span>
  -
  -
  -

* - Reproject/warp
  - `gdalwarp`
  - `gdal raster reproject`
  - {meth}`~geoutils.Raster.reproject`

* - Crop/clip
  - `gdal_translate` / `gdalwarp`
  - `gdal raster clip`
  - {meth}`~geoutils.Raster.crop` / {meth}`~geoutils.Raster.icrop`

* - Edit referencing
  - `gdal_edit` / `gdalmove.py`
  - `gdal raster edit`
  - {meth}`~geoutils.Raster.set_crs`, {meth}`~geoutils.Raster.set_transform`, {meth}`~geoutils.Raster.set_nodata`, {meth}`~geoutils.Raster.translate`

* - Convert file format
  - `gdal_translate`
  - `gdal raster convert`
  - {meth}`~geoutils.Raster.to_file`

* - Filter
  - —
  - `gdal raster neighbors`
  - {meth}`~geoutils.Raster.filter`

* - Proximity distance
  - `gdal_proximity`
  - `gdal raster proximity`
  - {meth}`~geoutils.Raster.proximity`
  
* - Raster calculator
  - `gdal_calc.py`
  - `gdal raster calc`
  - NumPy array interface on {attr}`~geoutils.Raster.data`)

* - Mosaic / merge rasters
  - `gdal_merge.py`
  - `gdalbuildvrt`+`gdal raster reproject`
  - {func}`~geoutils.raster.merge_rasters`
  
* - Stack rasters into multiband raster
  - `gdal_stack.py`
  - `gdal raster stack`
  - {func}`~geoutils.raster.stack_rasters`

* - Fill nodata gaps
  - `gdal_fillnodata.py`
  - `gdal raster fill-nodata`
  - Not implemented (planned)

* - Remove small raster regions
  - `gdal_sieve.py`
  - `gdal raster sieve`
  - Not implemented (planned)

* - Generate contours
  - `gdal_contour`
  - `gdal raster contour`
  - Not implemented

* - <span class="gu-table-section">Raster ⟶ Point</span>
  -
  -
  -

* - Interpolate at coordinates
  - `gdallocationinfo`
  - `gdal raster pixel-info`
  - {meth}`~geoutils.Raster.to_pointcloud`/{meth}`~geoutils.Raster.interp_points`

* - <span class="gu-table-section">Raster ⟶ Vector</span>
  -
  -
  -

* - Polygonize
  - `gdal_polygonize.py`
  - `gdal raster polygonize`
  - {meth}`~geoutils.Raster.polygonize`

* - <span class="gu-table-section">Vector ⟶ Vector</span>
  -
  -
  -

* - Reproject
  - `ogr2ogr`
  - `gdal vector reproject`
  - {meth}`~geoutils.Vector.reproject`

* - Crop/clip
  - `ogr2ogr -clipsrc`
  - `gdal vector clip`
  - {meth}`~geoutils.Vector.crop`

* - Translate
  - `ogr2ogr` (SQL transform)
  - —
  - {meth}`~geoutils.Vector.translate`

* - Copy
  - `ogr2ogr`
  - `gdal vector convert`
  - {meth}`~geoutils.Vector.copy`
  
* - Geometric operations
  - `ogr2ogr` (specific)
  - `gdal vector simplify/buffer/...`
  - {meth}`~geoutils.Vector.simplify` / {meth}`~geoutils.Vector.buffer` / ...

* - <span class="gu-table-section">Vector ⟶ Raster</span>
  -
  -
  -

* - Rasterize
  - `gdal_rasterize`
  - `gdal vector rasterize`
  - {meth}`~geoutils.Vector.rasterize`

* - <span class="gu-table-section">Point ⟶ Raster</span>
  -
  -
  -

* - Grid points
  - `gdal_grid`
  - `gdal vector grid`
  - {meth}`~geoutils.PointCloud.grid`
```

## GDAL DEM utilities

The table below maps common **GDAL DEM analysis utilities** (historically provided by `gdaldem`) to their equivalents in **xDEM**.

```{list-table}
:header-rows: 1
:widths: 2 3 3 2
:align: left
:class: tight-table

* - Operation
  - Old GDAL
  - New GDAL CLI
  - xDEM

* - <span class="gu-table-section">Terrain attributes</span>
  -
  -
  -

* - Slope
  - `gdaldem slope`
  - `gdal raster slope`
  - {func}`~xdem.DEM.slope`

* - Aspect
  - `gdaldem aspect`
  - `gdal raster aspect`
  - {func}`~xdem.DEM.aspect`

* - Hillshade
  - `gdaldem hillshade`
  - `gdal raster hillshade`
  - {func}`~xdem.DEM.hillshade`

* - Terrain Ruggedness Index
  - `gdaldem TRI`
  - `gdal raster TRI`
  - {func}`~xdem.DEM.terrain_ruggedness_index`

* - Topographic Position Index
  - `gdaldem TPI`
  - `gdal raster TPI`
  - {func}`~xdem.DEM.topographic_position_index`

* - Roughness
  - `gdaldem roughness`
  - `gdal raster roughness`
  - {func}`~xdem.DEM.roughness`
```