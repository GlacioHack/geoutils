(api)=
# API reference

This page provides a summary of GeoUtils’ API.
For more details and examples, refer to the relevant chapters in the main part of the
documentation.

```{eval-rst}
.. currentmodule:: geoutils
```

## Raster

```{eval-rst}
.. minigallery:: geoutils.Raster
      :add-heading:
```

### Opening a file

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Raster
    Raster.info
```

### Create from an array

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Raster.from_array
```

(api-raster-attrs)=

### Main attributes

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Raster.data
    Raster.crs
    Raster.transform
    Raster.nodata
    Raster.area_or_point
```

### Derived attributes

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Raster.shape
    Raster.height
    Raster.width
    Raster.count
    Raster.bands
    Raster.res
    Raster.bounds
    Raster.dtype
```

### Other attributes

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Raster.is_mask
    Raster.count_on_disk
    Raster.bands_on_disk
    Raster.is_loaded
    Raster.is_modified
    Raster.name
    Raster.driver
    Raster.tags
```

(api-geo-handle)=

### Geospatial handling methods

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Raster.crop
    Raster.icrop
    Raster.reproject
    Raster.polygonize
    Raster.proximity
    Raster.interp_points
    Raster.reduce_points
    Raster.filter
```

### Plotting

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Raster.plot
```

### Get statistics

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Raster.get_stats
```

### Get or update data methods

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Raster.copy
    Raster.astype
    Raster.set_mask
    Raster.set_nodata
    Raster.get_nanarray
    Raster.get_mask
    Raster.subsample
```

### I/O methods

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Raster.load
    Raster.to_file
    Raster.to_pointcloud
    Raster.from_pointcloud_regular
    Raster.to_rio_dataset
    Raster.to_xarray
```

### Coordinate and extent methods

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Raster.xy2ij
    Raster.ij2xy
    Raster.coords
    Raster.translate
    Raster.outside_image
```

### Projection methods

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Raster.get_metric_crs
    Raster.get_bounds_projected
    Raster.get_footprint_projected
    Raster.intersection
```

### Testing methods

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Raster.raster_equal
    Raster.georeferenced_grid_equal
```

### Arithmetic with other rasters, arrays or numbers

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Raster.__add__
    Raster.__sub__
    Raster.__neg__
    Raster.__mul__
    Raster.__truediv__
    Raster.__floordiv__
    Raster.__mod__
    Raster.__pow__
```

And reverse operations.

### Logical operators casting to mask (boolean raster)

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Raster.__eq__
    Raster.__ne__
    Raster.__lt__
    Raster.__le__
    Raster.__gt__
    Raster.__ge__
```

### Array interface with NumPy

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Raster.__array_ufunc__
    Raster.__array_function__
```

## Multiple rasters

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    raster.load_multiple_rasters
    raster.stack_rasters
    raster.merge_rasters
```

[//]: # (## Multiprocessing)

[//]: # ()
[//]: # (```{eval-rst})

[//]: # (.. autosummary::)

[//]: # (    :toctree: gen_modules/)

[//]: # ()
[//]: # (    raster.MultiprocConfig)

[//]: # ()
[//]: # (```)

## Vector

```{eval-rst}
.. minigallery:: geoutils.Vector
      :add-heading:
```

### Opening a file

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Vector
    Vector.info
```

### Main attributes

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Vector.ds
    Vector.crs
    Vector.bounds
    Vector.name
```

```{caution}
The {attr}`~geoutils.Vector.bounds` attribute of a {class}`~geoutils.Vector` corresponds to the {attr}`~geopandas.GeoDataFrame.total_bounds` attribute of a
{class}`~geopandas.GeoDataFrame`, for consistency between rasters and vectors (and can also be accessed through {attr}`~geoutils.Vector.total_bounds`).

The equivalent of {attr}`geopandas.GeoDataFrame.bounds` (i.e., a per-feature bounds) for {class}`Vectors<geoutils.Vector>` is {attr}`~geoutils.Vector.geom_bounds`.
```

### Geospatial handling methods

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Vector.crop
    Vector.reproject
    Vector.rasterize
    Vector.proximity
```

### Plotting

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Vector.plot
```

### Create mask

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Vector.create_mask
```

### Geometry manipulation

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Vector.buffer_metric
    Vector.buffer_without_overlap
```

### Projection tools

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Vector.get_metric_crs
    Vector.from_bounds_projected
    Vector.get_bounds_projected
    Vector.get_footprint_projected
```

### Indexing

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Vector.__getitem__
```

(vector-from-geopandas)=

### From Shapely and GeoPandas

#### Geometric attributes and methods

This first category of attributes and methods return a geometric output converted to a {class}`~geoutils.Vector` by default.

**Attributes:**

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Vector.boundary
    Vector.centroid
    Vector.convex_hull
    Vector.envelope
    Vector.exterior
```


**Methods:**

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Vector.representative_point
    Vector.normalize
    Vector.make_valid
    Vector.difference
    Vector.symmetric_difference
    Vector.union
    Vector.union_all
    Vector.intersection
    Vector.intersection_all
    Vector.clip_by_rect
    Vector.buffer
    Vector.simplify
    Vector.affine_transform
    Vector.translate
    Vector.rotate
    Vector.scale
    Vector.skew
    Vector.concave_hull
    Vector.delaunay_triangles
    Vector.voronoi_polygons
    Vector.minimum_rotated_rectangle
    Vector.minimum_bounding_circle
    Vector.extract_unique_points
    Vector.remove_repeated_points
    Vector.offset_curve
    Vector.reverse
    Vector.segmentize
    Vector.polygonize
    Vector.transform
    Vector.force_2d
    Vector.force_3d
    Vector.line_merge
    Vector.shortest_line
    Vector.interpolate
    Vector.shared_paths
    Vector.dissolve
    Vector.explode
    Vector.sjoin
    Vector.sjoin_nearest
    Vector.overlay
    Vector.clip
    Vector.snap
    Vector.to_crs
    Vector.set_crs
    Vector.get_geometry
    Vector.set_geometry
    Vector.rename_geometry
    Vector.set_precision
    Vector.get_precision
    Vector.get_coordinates
    Vector.cx
```

#### Non-geometric per-feature attributes and methods

This second category of attributes and methods return a non-geometric output with same length as the number of features. They are thus appended in the
dataframe of the current {class}`~geoutils.Vector` by default, using as column name the name of the operation (e.g., "area", "contains" or "intersects").

Otherwise, calling the method from {attr}`Vector.ds<geoutils.Vector.ds>`, they return a {class}`pandas.Series` as in GeoPandas.

**Attributes:**

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Vector.area
    Vector.length
    Vector.interiors
    Vector.geom_type
    Vector.geom_bounds
    Vector.is_valid
    Vector.is_empty
    Vector.is_ring
    Vector.is_simple
    Vector.is_ccw
    Vector.is_closed
    Vector.has_z
```

**Methods:**

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Vector.contains
    Vector.geom_equals
    Vector.crosses
    Vector.disjoint
    Vector.intersects
    Vector.overlaps
    Vector.touches
    Vector.within
    Vector.covers
    Vector.covered_by
    Vector.distance
    Vector.is_valid_reason
    Vector.count_coordinates
    Vector.count_geometries
    Vector.count_interior_rings
    Vector.get_precision
    Vector.minimum_clearance
    Vector.minimum_bounding_radius
    Vector.contains_properly
    Vector.dwithin
    Vector.hausdorff_distance
    Vector.frechet_distance
    Vector.hilbert_distance
    Vector.relate
    Vector.relate_pattern
    Vector.project
```


#### I/O, conversions and others

```{important}
The behaviour of methods below is not modified in {class}`~geoutils.Vector`, as they deal with outputs of different types.
To ensure those are up-to-date with GeoPandas, alternatively call those from {attr}`Vector.ds<geoutils.Vector.ds>`.
```

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Vector.from_file
    Vector.from_features
    Vector.from_postgis
    Vector.from_dict
    Vector.from_arrow
    Vector.to_file
    Vector.to_feather
    Vector.to_parquet
    Vector.to_arrow
    Vector.to_wkt
    Vector.to_wkb
    Vector.to_json
    Vector.to_postgis
    Vector.to_geo_dict
    Vector.to_csv
```


#### Other attributes and methods


```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Vector.has_sindex
    Vector.sindex
    Vector.total_bounds
```

```{seealso}
The methods above are described in [GeoPandas GeoSeries's API](https://geopandas.org/en/stable/docs/reference/geoseries.html) and [Shapely object's
documentation](https://shapely.readthedocs.io/en/stable/properties.html).
```

## Point cloud

```{eval-rst}
.. minigallery:: geoutils.PointCloud
      :add-heading:
```

### Opening a file

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    PointCloud
    PointCloud.info
```

### Main attributes

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    PointCloud.ds
    PointCloud.data_column
    PointCloud.data
    PointCloud.crs
```

### Other attributes

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    PointCloud.point_count
```


### Create and convert from data

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    PointCloud.from_xyz
    PointCloud.from_array
    PointCloud.from_tuples
    PointCloud.to_xyz
    PointCloud.to_array
    PointCloud.to_tuples

```

### Geospatial

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    PointCloud.crop
    PointCloud.reproject
    PointCloud.translate
```

### Interface with raster

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    PointCloud.grid
```

### Statistics

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    PointCloud.get_stats
    PointCloud.subsample
```

### Testing methods

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    PointCloud.pointcloud_equal
    PointCloud.georeferenced_coords_equal
```
