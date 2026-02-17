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

GeoUtils uses and proposes several data examples to manipulate and test the different features.

## Description

Several sites (Coromandel Peninsula in New Zealand, Mount Everest and Exploradores Glacier  in Chili) are proposed to cover different kinds of data:

| Alias                          |                      Site and Filename                       |    Type     |                                     Description                                     |
|-------------------------------:|:------------------------------------------------------------:|:-----------:|:-----------------------------------------------------------------------------------:|
| `"coromandel_lidar"`           |               Coromandel_Lidar<br/>points.laz                | Point Cloud |                    Land elevation over the area measured in 2021                    |
| `"everest_landsat_b4"`         |       Everest_Landsa<br/>LE71400412000304SGS00_B4.tif        |   Raster    |                  B04 (red) image of the Mount Everest area in 2000                  |
| `"everest_landsat_b4_cropped"` |   Everest_Landsat<br/>LE71400412000304SGS00_B4_cropped.tif   |   Raster    |              B04 (red) cropped image of the Mount Everest area in 2000              |
| `"everest_landsat_rgb"`        |         Everest_Landsat<br/>LE71400412000304SGS00_RGB.tif         |   Raster    |                     RGB image of the Mount Everest area in 2000                     |
| `"everest_rgi_outlines"`       |      Everest_Landsat<br/>15_rgi60_glacier_outlines.gpkg       |   Vector    |    Glacier outlines around the Mount Everest around 2000 [(1)](#doi-references)     |
| `"exploradores_aster_dem"`     |    Exploradores_ASTER<br/>AST_L1A_00303182012144228_Z.tif     |   Raster    |                Land elevation of the Exploradores Glacier area 2012                 |
| `"exploradores_rgi_outlines"`  |     Exploradores_ASTER<br/>17_rgi60_glacier_outlines.gpkg     |   Vector    | Glacier outlines around the Exploradores Glacier around 2000 [(1)](#doi-references) |


```{note}
If you need more information about the data, you can read this [page](https://github.com/GlacioHack/geoutils-data/blob/main/README.md)
of the [geoutils-data repository project](https://github.com/GlacioHack/geoutils-data) where they are stored, described and referenced with their Digital Object Identifier:

1. DOI: [10.7265/N5-RGI-60](https://doi.org/10.7265/N5-RGI-60)
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
