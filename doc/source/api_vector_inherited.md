(vector-from-geopandas)=
# From Shapely and GeoPandas

The methods on this page are inherited by {class}`~geoutils.Vector`. With the
{class}`vct <geoutils.VectorAccessor>` interface, call them directly on the
{class}`~geopandas.GeoDataFrame`.

```{eval-rst}
.. currentmodule:: geoutils
```

## Geometric attributes and methods

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
    Vector.snap
    Vector.to_crs
    Vector.set_crs
    Vector.get_geometry
    Vector.set_geometry
    Vector.rename_geometry
    Vector.set_precision
    Vector.get_coordinates
    Vector.cx
```

## Non-geometric per-feature attributes and methods

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

## I/O, conversions and others

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

## Other attributes and methods

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
