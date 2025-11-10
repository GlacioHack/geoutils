# Release notes

Below, the release notes for all minor versions and our roadmap to a first major version.

## 0.2.0

GeoUtils version 0.2 is the **second minor release** since the creation of the project. It is the result of months of work to
consolidate the point cloud features towards a stable API that interfaces well with other objects. Parallel work on scalability
with Dask and Multiprocessing is ongoing, and should soon be released in a 0.3.

GeoUtils 0.2 adds:
- **A point cloud object** with its own specific methods (e.g, gridding), that can be used as match-reference for other operations (e.g., interpolating at points), and supports arithmetic (e.g., indexing, NumPy stats) and geometric (e.g., masking) functionalities with the same API as for `Raster` objects,
- **Preliminary statistics** features common to rasters and point clouds, which will be expanded with binning and spatial statistics.

A few changes might be required to adapt from previous versions:
- Specify `Raster.interp_points(as_array=True)` to mirror the previous behaviour of returning a 1D array of interpolated values, otherwise now returns a point cloud by default.
- The `Mask` class is deprecated in favor of `Raster(is_mask=True)` to declare a boolean-type Raster, but should keep working until 0.3.

## 0.1.0

GeoUtils version 0.1 is the **first minor release** since the creation of the project in 2020. It is the result of years of work
to consolidate and re-structure features into a mature and stable API to minimize future breaking changes.

**All the core features drafted at the start of the project are now supported**, and there is a **clear roadmap
towards a first major release 1.0**. This minor release also adds many tests and improves significantly the documentation
from the early-development state of the package.

The re-structuring created some breaking changes, though minor.

See details below, including **a guide to help migrate code from early-development versions**.

### Features

GeoUtils now gathers the following core features:
- **Geospatial data objects** core to quantatiative analysis, which are rasters, vectors and point cloud (preliminary) functionalities,
- **Referencing and transformations** using a consistent API with match-reference functionalities,
- **Raster–vector–point interface** to interface between the core objects, including rasterize and polygonize, interpolate and grid, and conversions,
- **Distance operations** for all objects.

(migrate-early)=
### Migrate from early versions

The following changes **might be required to solve breaking changes**, depending on your early-development version:
- Rename `.show()` to `.plot()` for all data objects,
- Rename `.dtypes` to `dtype` for `Raster` objects,
- Operations `.crop()`, `shift()` and `to_vcrs()` are not done in-place by default anymore, replace by `rst = rst.crop()` or `rst.crop(..., inplace=True)` to mirror the old default behaviour,
- Rename `.shift()` to `.translate()` for `Raster` objects,
- Several function arguments are renamed, in particular `dst_xxx` arguments of `.reproject()` are all renamed to `xxx` e.g. `dst_crs` to `crs`,
- New user warnings are sometimes raised, in particular if some metadata is not properly defined such as `.nodata`. Those should give an indication as how to silence them.

## Roadmap to 1.0

Based on recent and ongoing progress, we envision the following roadmap.

**Releases of 0.2, 0.3, 0.4, etc**, for the following planned (ongoing) additions:
- The **addition of a point cloud `PointCloud` data object**, inherited from the `Vector` object alongside many features at the interface of point and raster,
- The **addition of a Xarray accessor `rst`** mirroring the `Raster` object, to work natively with Xarray objects and add support on out-of-memory Dask operations for most of GeoUtils' features,
- The **addition of a GeoPandas accessor `pc`** mirroring the `PointCloud` object, to work natively with GeoPandas objects,
- The **addition of statistical features** including zonal statistics (e.g., statistics per vector geometry), grouped statistics (e.g., binning with other variables) and spatial statistics (variogram and kriging) through optional dependencies.
- The **addition of filtering and gap-filling features** natively robust to nodata and working similarly for all type of geospatial objects.

**Release of 1.0** once all these additions are fully implemented, and after feedback from the community.
