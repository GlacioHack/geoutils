(about-geoutils)=

# About GeoUtils

## What is GeoUtils?

GeoUtils<sup>1</sup> is a **[Python](https://www.python.org/) package for the handling and analysis of georeferenced data**, developed with the objective of
making such analysis accessible, efficient and reliable.

```{margin}
<sup>1</sup>With name standing for *Geospatial Utilities*.
```

In a few words, GeoUtils can be described as a **convenience package for end-users focusing on geospatial analysis**. It allows to write shorter
code through consistent higher-level operations, implicit object behaviour and interfacing. In addition, GeoUtils adds several analysis-oriented
functions that require many steps to perform with other packages, and which are robustly tested.

GeoUtils is designed for all Earth and planetary observation science. However, it is generally **most useful for surface applications that rely on
moderate- to high-resolution data** (requiring reprojection, re-gridding, point interpolation, and other types of fine-grid analysis).

## Why use GeoUtils?

GeoUtils is built on top of [Rasterio](https://rasterio.readthedocs.io/en/latest/), [GeoPandas](https://geopandas.org/en/stable/docs.html)
and [PyProj](https://pyproj4.github.io/pyproj/stable/index.html) for georeferenced operations, and relies on [NumPy](https://numpy.org/doc/stable/),
[SciPy](https://docs.scipy.org/doc/scipy/) and [Xarray](https://docs.xarray.dev/en/stable/) for scientific computing to provide:
- A **common and consistent framework** for efficient raster and vector handling,
- A structure following the **principal of least knowledge**<sup>2</sup> to foster accessibility,
- A **pythonic arithmetic** and **NumPy interfacing** for robust numerical computing.

```{margin}
<sup>2</sup>Or the [Law of Demeter](https://en.wikipedia.org/wiki/Law_of_Demeter) for software development.
```

In particular, GeoUtils:
- Rarely requires more than **single-line operations** thanks to its object-based structure,
- Strives to rely on **lazy operations** under-the-hood to avoid unnecessary data loading,
- Allows for **match-reference operations** to facilitate geospatial handling,
- Re-implements **several of [GDAL](https://gdal.org/)'s features** missing in other packages (e.g., proximity, gdalDEM),
- Naturally handles **different `dtypes` and `nodata`** values through its NumPy masked-array interface.


```{note}
More on these core features of GeoUtils in the {ref}`quick-start`, or {ref}`core-index` for details.
```

## Why the need for GeoUtils?

Recent community efforts have improved open-source geospatial analysis in Python, allowing to **move away from the low-level functions and
complexity of [GDAL and OGR](https://gdal.org/)'s Python bindings** for raster and vector handling. Those efforts include in particular
[Rasterio](https://rasterio.readthedocs.io/en/latest/) and [GeoPandas](https://geopandas.org/en/stable/docs.html).

However, these new packages still maintain a relatively low-level API to serve all types of geospatial informatics users, **slowing down end-users focusing
on data analysis**. As a result, basic interfacing between vectors and rasters is not always straightforward and simple higher-level operations (such as
reprojection to match a vector or raster reference, or point interpolation) are not always computed consistently in the community.

On one hand, [Rasterio](https://rasterio.readthedocs.io/en/latest/) focuses largely on reading, projecting and writing, and thus **requires
array extraction, re-encapsulation, and the volatile passing of metadata** either before, during or after any numerical calculations. On the other hand,
[GeoPandas](https://geopandas.org/en/stable/docs.html) focuses on integrating [Shapely](https://shapely.readthedocs.io/en/stable/) geometries in the
[Pandas](https://pandas.pydata.org/) framework, which is practical for tabular analysis but **yields a multitude of outputs (dataframes, series, geoseries,
geometries), often requiring object re-construction and specific reprojection routines** to analyze with other data, or derive metric attributes (area,
length).

Finally, **many common geospatial analysis tools are generally unavailable** in existing packages (e.g., boolean-masking from vectors,
[proximity](https://gdal.org/programs/gdal_proximity.html) estimation, metric buffering) as they rely on a combination of lower-level operations.

```{admonition} Conclusion
Having higher-level geospatial tools implemented in a **consistent** manner and tested for **robustness** is essential for the wider geospatial community.
```

## Side-by-side examples with Rasterio and GeoPandas

This first side-by-side example demonstrates the difference with Rasterio for opening a raster, reprojecting on 
another "reference" raster, performing array operations respectful of nodata values, and saving to file.


```{note}
**GeoUtils does not just wrap the Rasterio or GeoPandas operations showed below**. Instead, it defines **raster- and 
vector-centered objects to ensure consistent geospatial object behaviour that facilitates those operations** (e.g., by implicitly passing metadata, loading, or interfacing).
```

`````{list-table}
---
header-rows: 1
---
* - GeoUtils
  - Rasterio
* - ```python
    import geoutils as gu
    
    # Opening of two rasters
    rast1 = gu.Raster("myraster1.tif")
    rast2 = gu.Raster("myraster2.tif")
    
    # Reproject 1 to match 2
    # (raster 2 not loaded, only metadata)
    rast1_reproj = rast1.reproject(
        dst_ref = rast2
        )
    
    # Array interfacing and implicit loading
    # (raster 2 loads implicitly)
    rast_result = (1 + rast2) / rast1_reproj
    
    # Saving
    rast_result.save("myresult.tif")
    ```
    
  - ```python
    import rasterio as rio
    import numpy as np
    
    # Opening of two rasters
    rast1 = rio.io.DatasetReader("myraster1.tif")
    rast2 = rio.io.DatasetReader("myraster2.tif")
    
    # Equivalent of a match-reference reprojection
    # (returns an array, not a raster-type object)
    arr1_reproj, _ = rio.warp.reproject(
        source = rast1.read(), 
        destination = np.ones(rast2.shape), 
        src_transform = rast1.transform, 
        src_crs = rast1.crs, 
        src_nodata = rast1.nodata, 
        dst_transform = rast2.transform, 
        dst_crs = rast2.crs, 
        dst_nodata = rast2.nodata,
        )
    
    # Equivalent of array interfacing
    # (ensuring nodata and dtypes are rightly 
    # propagated through masked arrays)
    ma1_reproj = np.ma.MaskedArray(
        data = arr1_reproj, 
        mask = (arr1_reproj == rast2.nodata)
        )
    ma2 = rast2.read(masked = True)
    ma_result = (1 + ma2) / (ma1_reproj)
    
    # Equivalent of saving
    # (requires to define a logical
    # nodata for the data type)
    out_nodata = custom_func(
        dtype = ma_result.dtype,
        nodata1 = rast1.nodata,
        nodata2 = rast2.nodata
        )
    with rio.open(
        "myresult.tif",
        mode = "w",
        height = rast2.height,
        width = rast2.width,
        count = rast1.count,
        dtype = ma_result.dtype,
        crs = rast2.crs,
        transform = rast2.transform,
        nodata = rast2.nodata,
    ) as dst:
        dst.write(ma_result.filled(out_nodata))
    ```
`````

This second side-by-side example demonstrates the difference with GeoPandas (and Rasterio) for opening a vector, 
applying a metric geometric operation (buffering), rasterizing into a boolean mask, and indexing a raster with that mask.

`````{list-table}
---
header-rows: 1
---
* - GeoUtils
  - GeoPandas (and Rasterio)
* - ```python
    import geoutils as gu
    
    # Opening a vector and a raster
    vect = gu.Vector("myvector.shp")
    rast = gu.Raster("myraster.tif")
    
    # Metric buffering
    vect_buff = vect.buffer(distance = 100)
    
    # Create a mask on the raster grid
    # (raster not loaded, only metadata)
    mask = vect_buff.create_mask(
        dst_ref = rast
        )
    
    # Index raster values on mask
    # (raster loads implicitly)
    values = rast[mask]
    ```
    
  - ```python
    import geopandas as gpd
    import rasterio as rio
    
    # Opening a vector and a raster
    df = gpd.read_file("myvector.tif")
    rast2 = rio.io.DatasetReader("myraster.tif")
    
    # Equivalent of a metric buffering 
    # (while keeping a frame object)
    gs_m_crs = df.to_crs(df.estimate_utm_crs())
    gs_m_crs_buff = gs_m_crs.buffer(distance = 100)
    gs_buff = gs_m_crs_buff.to_crs(df.crs)
    df_buff = gpd.GeoDataFrame(
        geometry = gs_buff
        )
        
    # Equivalent of creating a rasterized mask
    # (ensuring CRS are similar)
    df_buff = df_buff.to_crs(rast2.crs)
    mask = features.rasterize(
        shapes = gdf.geometry, 
        fill = 0, 
        out_shape = rast2.shape, 
        transform = rast2.transform, 
        default_value = 1, 
        dtype = "uint8"
        )
    mask = mask.astype("bool")
    
    # Equivalent of indexing with mask
    values = rast2.read(1, masked = True)[mask]
`````