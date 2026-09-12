(benchmarking-performance)=
# Performance

GeoUtils benchmarks its functionalities to measure **execution time and memory usage**, check **lazy and out-of-core behaviour** with
Dask and Multiprocessing, track **improvements or regressions over time**, and compare different **execution modes** and
**calculation engines** against the [**GDAL CLI**](https://gdal.org/en/stable/programs/index.html). Chunked operations
also compare strategies used to coordinate selections or reconcile results across chunks.

The benchmarks use GeoUtils' profiling tool described in {ref}`profiling`, which relies on [**psutil**](https://psutil.readthedocs.io/) and [**Dask's worker memory diagnostics**](https://distributed.dask.org/en/stable/worker-memory.html).

## Improvements or regressions over time

The [**GeoUtils benchmark webpage**](https://glaciohack.github.io/geoutils/) provides performance of core functionalities and their changes with commit history.
It relies on [Airspeed Velocity (ASV)](https://asv.readthedocs.io/) to record reproducible performance measurements for each commit and publish them.

## Comparison across execution modes, engines and GDAL CLI

GeoUtils compares **eager, Dask and Multiprocessing** execution using the same deterministic inputs. Reprojection,
polygonization, rasterization and gridding additionally use equivalent **GDAL CLI commands** with matching
output grids, data types and storage options where possible. Every result identifies its operation, numerical method,
calculation engine, chunk strategy and execution mode separately. Engine comparisons use eager execution, while
execution-mode and chunk-strategy comparisons fix the other dimensions. Fixed-size Numba checks additionally exercise
each compiled gridding kernel in Dask and Multiprocessing workers. The Rasterio/GDAL calculation engine used inside
GeoUtils remains distinct from the external GDAL CLI reference.

Two reference graphics summarize the core results:

- **Execution time relative to GDAL** for each comparable operation (using end-to-end time with file reading/writing for comparison),
- **Peak memory usage as raster size increases**, including the full process and its workers.

:::{figure} https://glaciohack.github.io/geoutils/documentation/time_relative_to_gdal.svg
:alt: GeoUtils execution mode time relative to GDAL for four raster operations

Execution time on the largest raster size shared by every execution mode and the GDAL CLI. GDAL is the reference.
:::

:::{figure} https://glaciohack.github.io/geoutils/documentation/peak_ram_by_raster_size.svg
:alt: Peak memory usage by raster size for GeoUtils execution modes and GDAL

Peak memory usage for the benchmark process and all execution-mode workers as raster dimensions increase.
:::

These graphics show the latest complete CI benchmark and may be newer than this documentation version.
[Download the exact values and run metadata](https://glaciohack.github.io/geoutils/documentation/benchmark_snapshot.json).

## Test suite for scalable execution and large datasets

Every GeoUtils operation listed as {ref}`chunked or lazy <scalability-support>` is tested
on all supported Python versions and operating systems to ensure its respect of Dask laziness, deferred I/O, and loading behaviour,
while yielding **exactly** the same output as in-memory operations.

In addition, GeoUtils also includes large data tests running on latest Python and Ubuntu releases,
verifying that these operations use less memory than loading the full raster would require, while yielding a correct result.
