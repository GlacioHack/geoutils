(about-geoutils)=

# About GeoUtils

Prefer to **grasp GeoUtils' core concepts by comparing with other Python packages**? Further below is a **{ref}`side-by-side code comparison with Rasterio and GeoPandas<comparison-rasterio-geopandas>`**.

## What is GeoUtils?

GeoUtils<sup>1</sup> is a **[Python](https://www.python.org/) package for the analysis of georeferenced data**, developed with the objective of
making such analysis accessible, efficient and reliable.

```{margin}
<sup>1</sup>With name standing for *Geospatial Utilities*.
```

In a few words, GeoUtils can be described as a **convenience package for end-users focusing on geospatial analysis**. It allows to write shorter
code through consistent higher-level operations, implicit object behaviour and interfacing. In addition, GeoUtils adds several analysis-oriented
functions that require many steps to perform with other packages, and which are robustly tested.

GeoUtils is designed for all Earth and planetary observation science. However, it is generally **most useful for surface applications that rely on
moderate- to high-resolution data** (requiring reprojection, re-gridding, point interpolation, and other types of fine-grid analysis).

GeoUtils is built on top of [Rasterio](https://rasterio.readthedocs.io/en/latest/), [GeoPandas](https://geopandas.org/en/stable/docs.html)
and [PyProj](https://pyproj4.github.io/pyproj/stable/index.html) for georeferenced operations, and relies on [NumPy](https://numpy.org/doc/stable/),
[SciPy](https://docs.scipy.org/doc/scipy/) and [Xarray](https://docs.xarray.dev/en/stable/) for scientific computing to provide:
- A **common and consistent framework** for efficient raster, vector and point data handling,
- A structure following the **principal of least knowledge**<sup>2</sup> to foster accessibility,
- A **pythonic arithmetic** and **NumPy interfacing** for numerical computing.

```{margin}
<sup>2</sup>Or the [Law of Demeter](https://en.wikipedia.org/wiki/Law_of_Demeter) for software development.
```

In particular, GeoUtils:
- Rarely requires more than **single-line operations** thanks to its object-based structure,
- Strives to rely on **lazy operations** under-the-hood to avoid unnecessary data loading,
- Allows for **match-reference operations** to facilitate geospatial handling,
- Re-implements **several of [GDAL](https://gdal.org/)'s features** missing in other packages (e.g., proximity, gdalDEM),
- Naturally handles **different `dtype` and `nodata`** values through its NumPy masked-array interface.

```{note}
We are working on adding Dask support through an Xarray accessor for our next release.
```

## Who is behind GeoUtils?

GeoUtils was created by a group of researchers with expertise in geospatial data analysis for the cryosphere.
Nowadays, its development is **jointly led by researchers in geospatial analysis** (including funding from NASA and SNSF) **and
engineers from CNES** (French Space Agency).

Most contributors and users are scientists or industrials working in **various fields of Earth observation**.

::::{grid}
:reverse:

:::{grid-item}
:columns: 4
:child-align: center

```{image} ./_static/nasa_logo.svg
    :width: 200px
    :class: dark-light
```

:::

:::{grid-item}
:columns: 4
:child-align: center

```{image} ./_static/snsf_logo.svg
    :width: 220px
    :class: only-light
```

```{image} ./_static/snsf_logo_dark.svg
    :width: 220px
    :class: only-dark
```

:::

:::{grid-item}
:columns: 4
:child-align: center

```{image} ./_static/cnes_logo.svg
    :width: 200px
    :class: only-light
```

```{image} ./_static/cnes_logo_dark.svg
    :width: 200px
    :class: only-dark
```

:::


::::

More details about the people behind GeoUtils, funding sources, and the package's objectives can be found on the **{ref}`credits` pages**.

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

(comparison-rasterio-geopandas)=
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
* - ```{eval-rst}
    .. literalinclude:: code/about_geoutils_sidebyside_raster_geoutils.py
        :language: python
        :lines: 16-31
    ```

  - ```{eval-rst}
    .. literalinclude:: code/about_geoutils_sidebyside_raster_rasterio.py
        :language: python
        :lines: 11-58
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
* - ```{eval-rst}
    .. literalinclude:: code/about_geoutils_sidebyside_vector_geoutils.py
        :language: python
        :lines: 11-26
    ```

  - ```{eval-rst}
    .. literalinclude:: code/about_geoutils_sidebyside_vector_geopandas.py
        :language: python
        :lines: 11-34
    ```
`````
