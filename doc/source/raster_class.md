(raster-class)=

# The georeferenced raster (`Raster`)

## Object definition

A {class}`geoutils.Raster` is a georeferenced raster read, written or reprojected by `rasterio`.

It contains:
- an array `.data` as a {class}`numpy.ma.MaskedArray`,
- a geotransform `.transform` as a {class}`affine.Affine`,
- a coordinate reference system (CRS) `.crs` as a {class}`pyproj.CRS`,
- a nodata value `.nodata` as ``float`` or ``int``.


## Open and save

```{literalinclude} code/raster-basics_open_file.py
:lines: 2-8
```

````{margin} 
**Example of raster from {func}`~geoutils.Raster.info`:**

```{literalinclude} code/index_rast_info.py
    :lines: 11-12
    :language: python
```

```{eval-rst}
.. program-output:: $PYTHON code/index_rast_info.py
        :shell:
```

````

To print information directly to your console:

```python
print(image)
```

```{eval-rst}
.. program-output:: $PYTHON -c "exec(open('code/raster-basics_open_file.py').read()); print(image)"
        :shell:
```

If you'd like to retrieve a string of information about the raster to be saved
to a variable, output to a text file etc:

```{literalinclude} code/raster-basics_open_file.py
:lines: 10
```

With added stats:

```{literalinclude} code/raster-basics_open_file.py
:lines: 12
```

Then to write a file:

```{literalinclude} code/raster-basics_open_file.py
:lines: 14-15
```

Or just print nicely to console:

```python
print(information)
```

```{eval-rst}
.. program-output:: $PYTHON -c "exec(open('code/raster-basics_open_file.py').read()); print(information)"
        :shell:
```

## Arithmetic

## Array interface

## Crop

Comparing multiple rasters can often be a burden if multiple coordinate systems, bounding boxes, and resolutions are involved.
The {class}`geoutils.Raster` class simplifies this using two methods: `Raster.crop()` and `Raster.reproject()`.

{func}`geoutils.Raster.crop`

If a large raster should be cropped to a smaller extent without changing the uncropped data, this is possible through the crop function.

```{literalinclude} code/raster-basics_cropping_and_reprojecting.py
:lines: 4-6
```

```python
print(large_image.shape)

print(smaller_image.shape)
```

prints:

```{eval-rst}
.. program-output:: $PYTHON -c "exec(open('code/raster-basics_cropping_and_reprojecting.py').read());print(large_image_orig.shape);print(smaller_image.shape)"
        :shell:
```

If we want to crop the larger image to fit the smaller one:

```{literalinclude} code/raster-basics_cropping_and_reprojecting.py
:lines: 9
```

Now, they have the same shape, and can be compared directly:

```python
print(large_image.shape)

print(smaller_image.shape)
```

prints:

```{eval-rst}
.. program-output:: $PYTHON -c "exec(open('code/raster-basics_cropping_and_reprojecting.py').read());print(large_image.shape);print(smaller_image.shape)"
        :shell:
```

## Reproject

{func}`geoutils.Raster.reproject`

For rasters with different coordinate systems, resolutions or grids, reprojection is needed to fit one raster to another.
`Raster.reproject()` is apt for these use-cases:

```{literalinclude} code/raster-basics_cropping_and_reprojecting.py
:lines: 11
```

This call will crop, project and resample the `larger_raster` to fit the `smaller_raster` exactly.
By default, `Raster.resample()` uses nearest neighbour resampling, which is good for fast reprojections, but may induce unintended artefacts when precision is important.
It is therefore recommended to choose the method that fits the purpose best, using the `resampling=` keyword argument:

1. `resampling="nearest"`: Default. Performant but is not good for changes in resolution and grid.
2. `resampling="bilinear"`: Good when changes in resolution and grid are involved.
3. `resampling="cubic_spline"`: Often considered the best approach. Not as performant as simpler methods.

All valid resampling methods can be seen in the [Rasterio documentation](https://rasterio.readthedocs.io/en/latest/api/rasterio.enums.html#rasterio.enums.Resampling).

```{eval-rst}
.. minigallery:: geoutils.Raster
        :add-heading:
        :heading-level: -
```

## Polygonize

## Proximity

## Interpolate or extract to point

## Export