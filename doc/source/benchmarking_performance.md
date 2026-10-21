(benchmarking-performance)=
# Performance

GeoUtils benchmarks its operations to ensure **efficiency and robustness to updates**.

The benchmarks tests measure **execution time and memory usage**, check **lazy and out-of-core behaviour** with
Dask and Multiprocessing, track **improvements or regressions over time**, and compare different **execution modes** and
**calculation engines**, including external checks against [**GDAL CLI**](https://gdal.org/en/stable/programs/index.html) and PDAL.

The benchmarks use GeoUtils' profiling tool described in {ref}`profiling`.

## Improvements or regressions over time

The [**GeoUtils benchmark webpage**](https://glaciohack.github.io/geoutils/) summarizes performance of core functionalities and their changes with commit history.
It relies on [Airspeed Velocity (ASV)](https://asv.readthedocs.io/) to record reproducible performance measurements for each commit and publish them.

## Comparison across execution modes, engines and external CLI tools

GeoUtils compares **eager, Dask and Multiprocessing** execution using the same deterministic inputs. Reprojection,
clipping, polygonization, rasterization and gridding additionally use equivalent **GDAL CLI commands** with matching
output grids, data types and storage options where possible. Every result identifies its operation, numerical method,
calculation engine, chunk strategy and execution mode separately. Engine comparisons use eager execution, while
execution-mode and chunk-strategy comparisons fix the other dimensions. Fixed-size Numba checks additionally exercise
each compiled gridding kernel in Dask and Multiprocessing workers. The Rasterio/GDAL calculation engine used inside
GeoUtils remains distinct from the external GDAL CLI reference. Raster subsampling and raster-to-point conversion use
equivalent **PDAL CLI pipelines** as an additional external reference.

Four reference graphics summarize the core results:

- **End-to-end time relative to GDAL** for each comparable operation
- **Additional RAM by operation** after subtracting each initialized process-tree baseline
- **Additional RAM as raster size increases**
- **End-to-end time relative to PDAL** for raster subsampling and point cloud conversion

:::{figure} https://glaciohack.github.io/geoutils/documentation/time_relative_to_gdal.svg
:alt: End-to-end GeoUtils execution-mode time relative to GDAL for five raster operations

Execution time on the largest raster size shared by every execution mode and the GDAL CLI. GDAL is the reference.
:::

:::{figure} https://glaciohack.github.io/geoutils/documentation/memory_increase_by_operation.svg
:alt: Additional process-tree RAM by operation for GeoUtils execution modes and GDAL

Additional RAM on the largest shared raster. Each bar is the process-tree peak minus the initialized baseline measured
immediately before the operation.
:::

:::{figure} https://glaciohack.github.io/geoutils/documentation/memory_increase_by_raster_size.svg
:alt: Additional process-tree RAM by raster size for GeoUtils execution modes and GDAL

Additional RAM for the benchmark process and all execution-mode workers as raster dimensions increase.
:::

:::{figure} https://glaciohack.github.io/geoutils/documentation/time_relative_to_pdal.svg
:alt: End-to-end GeoUtils execution-mode time relative to PDAL for raster point operations

End-to-end time on the largest workload shared by GeoUtils and PDAL. PDAL is the reference at one. Dask completes an
in-memory dataframe, while Multiprocessing and PDAL complete file-backed GeoPackage output.
:::

These graphics show the latest complete CI benchmark and may be newer than this documentation version.
[Download the exact values and run metadata](https://glaciohack.github.io/geoutils/documentation/benchmark_snapshot.json).

## Test suite for scalable execution and large datasets

Every GeoUtils operation listed as {ref}`chunked or lazy <scalability-support>` is tested
on all supported Python versions and operating systems to ensure its respect of Dask laziness, deferred I/O, and loading behaviour,
while yielding **exactly** the same output as in-memory operations, within numerical tolerances.

In addition, GeoUtils also includes large data tests running on latest Python and Ubuntu releases,
verifying that these operations use less memory than loading the full raster would require, while yielding a correct result.
