(gapfill)=
# Gap-filling

Gap-filling **estimates missing raster values from nearby valid cells**.

{meth}`ds.rst.fill_nodata() or Raster.fill_nodata() <RasterBase.fill_nodata>`

Use `resampling="idw"` (default) for inverse-distance interpolation, or `"nearest"` to copy a nearby value.
`max_search_distance` limits the search in **pixels**; cells beyond that distance remain missing.
The result preserves the raster grid and georeferencing.

This method processes a **complete raster grid in memory**. Crop to the area of interest before filling gaps
when the full grid is too large. See {ref}`scalability-support` for execution limits.
