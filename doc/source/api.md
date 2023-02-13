(api)=
# API reference

This page provides an auto-generated summary of GeoUtils’s API. 
For more details and examples, refer to the relevant chapters in the main part of the 
documentation.

```{eval-rst}
.. currentmodule:: geoutils
```

**Overview of class inheritance in GeoUtils:**

```{eval-rst}
.. inheritance-diagram:: geoutils
        :top-classes: geoutils.Raster geoutils.Vector
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

### Unique attributes

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Raster.data
    Raster.crs
    Raster.transform
    Raster.nodata
```

### Derived attributes

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Raster.shape
    Raster.height
    Raster.width
    Raster.count
    Raster.count_on_disk
    Raster.indexes
    Raster.indexes_on_disk
    Raster.res
    Raster.bounds
    Raster.dtypes
    Raster.is_loaded
    Raster.is_modified
    Raster.name
    Raster.driver
```

### Geospatial handling methods

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Raster.crop
    Raster.reproject
    Raster.polygonize
    Raster.proximity
    Raster.value_at_coords
    Raster.interp_points
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
```
    
### I/O methods

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Raster.load
    Raster.save
    Raster.to_points
    Raster.to_rio_dataset
    Raster.to_xarray
```

### Logical methods

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Raster.raster_equal
    Raster.equal_georeferenced_grid
```

### Coordinate and extent methods

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Raster.xy2ij
    Raster.ij2xy
    Raster.coords
    Raster.shift
    Raster.get_bounds_projected
    Raster.intersection
    Raster.outside_image   
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
    Raster.__power__    
```

And reverse operations.

### Logical operator casting to Mask

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Raster.__eq__
    Raster.__neq__
    Raster.__lt__
    Raster.__le__
    Raster.__gt__
    Raster.__ge__
    Raster.__mod__
    Raster.__power__    
```

### Array interface with NumPy

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Raster.__array_ufunc__
    Raster.__array_function__  
```

## SatelliteImage

```{eval-rst}
.. minigallery:: geoutils.SatelliteImage
      :add-heading:
```

### Opening a file

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    SatelliteImage
```

### Satellite image metadata

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    SatelliteImage.datetime
    SatelliteImage.tile_name
    SatelliteImage.satellite
    SatelliteImage.sensor
    SatelliteImage.product
    SatelliteImage.version
```

## Mask

```{eval-rst}
.. minigallery:: geoutils.Mask
      :add-heading:
```

### Opening a file

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Mask
```

### Booleans

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Mask.datetime
```

## Vectors


### Opening a file

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Vector
    Vector.info
```

### Unique attributes

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/

    Vector.ds
    Vector.crs
    Vector.bounds
    Vector.name
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

### Coordinate and extent methods

```{eval-rst}
.. autosummary::
    :toctree: gen_modules/
    
    Vector.get_bounds_projected 