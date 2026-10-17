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
(data)=

# Data examples

GeoUtils relies on **several data examples** to manipulate and test its different features.

## Description

Three sites (Coromandel Peninsula in New Zealand, Mount Everest and Exploradores Glacier in Chile) cover different kinds of data:

```{list-table}
:header-rows: 1
:widths: 34 14 52
:align: left

* - Alias
  - Type
  - Description
* - `coromandel_lidar`
  - Point cloud
  - Land elevation over the area measured in 2021 [(1)](#doi)
* - `everest_landsat_b4`
  - Raster
  - Band 4 (near infrared) image of the Mount Everest area in 2000
* - `everest_landsat_b4_cropped`
  - Raster
  - Band 4 (near infrared) cropped image of the Mount Everest area in 2000
* - `everest_landsat_rgb`
  - Raster
  - RGB image of the Mount Everest area in 2000
* - `everest_rgi_outlines`
  - Vector
  - Glacier outlines around the Mount Everest around 2000 [(2)](#doi)
* - `exploradores_aster_dem`
  - Raster
  - Land elevation of the Exploradores Glacier area 2012 [(3)](#doi)
* - `exploradores_rgi_outlines`
  - Vector
  - Glacier outlines around the Exploradores Glacier around 2000 [(2)](#doi)
```

:::{dropdown} Source filenames

**Coromandel_Lidar**

- `coromandel_lidar`: `points.laz`

**Everest_Landsat**

- `everest_landsat_b4`: `LE71400412000304SGS00_B4.tif`
- `everest_landsat_b4_cropped`: `LE71400412000304SGS00_B4_cropped.tif`
- `everest_landsat_rgb`: `LE71400412000304SGS00_RGB.tif`
- `everest_rgi_outlines`: `15_rgi60_glacier_outlines.gpkg`

**Exploradores_ASTER**

- `exploradores_aster_dem`: `AST_L1A_00303182012144228_Z.tif`
- `exploradores_rgi_outlines`: `17_rgi60_glacier_outlines.gpkg`
:::

(doi)=
```{note}
For more information about the data, refer to the [GitHub page of geoutils-data](https://github.com/GlacioHack/geoutils-data/blob/main/README.md)
 where they are stored, described and referenced, along with their digital object identifiers (DOIs):

1. New Zealand lidar: [10.5069/G91J980G](https://doi.org/10.5069/G91J980G)
2. Randolph Glacier Inventory: [10.7265/N5-RGI-60](https://doi.org/10.7265/N5-RGI-60)
3. ASTER L1A imagery: [10.5067/ASTER/AST_L1A.003](https://doi.org/10.5067/ASTER/AST_L1A.003)
```

## Access to data

### Python

If you want to use one of the example data, you can run this function with the corresponding data alias:

```
import geoutils as gu

# Download the 2012 raster DEM from Exploradores_ASTER dataset and return its path
path = gu.examples.get_path("exploradores_aster_dem")
```

It downloads the entire dataset if it was not already available and returns its absolute file path.

### Bash

To download the data samples, you can run:

```bash
mkdir data_examples
tar -xvz -C data_examples --strip-components 2 -f <(wget -q -O - https://github.com/GlacioHack/geoutils-data/archive/e758274647a8dd2656d73c3026c90cc77cab8a86.tar.gz)
```
