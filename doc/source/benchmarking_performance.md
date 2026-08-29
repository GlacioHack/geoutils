(benchmarking-performance)=
# Performance

GeoUtils benchmarks its functionalities to measure **RAM and wall time**, check **lazy and out-of-core behaviour** with
Dask and Multiprocessing, track **improvements or regressions over time**, and compare different **execution modes** and
**calculation engines** against the [**GDAL CLI**](https://gdal.org/en/stable/programs/index.html). Chunked operations
also compare strategies used to coordinate selections or reconcile results across chunks.

The benchmarks use GeoUtils' profiling tool described in {ref}`profiling`, which relies on [**psutil**](https://psutil.readthedocs.io/) and [**Dask's worker memory diagnostics**](https://distributed.dask.org/en/stable/worker-memory.html).

## Improvements or regressions over time

The [**GeoUtils benchmark webpage**](https://glaciohack.github.io/geoutils/) provides performance of core functionalities and their changes with commit history.
It relies on [Airspeed Velocity (ASV)](https://asv.readthedocs.io/) to record fixed performance measurements for each commit and publish them.

## Comparison across execution modes, engines and GDAL CLI

GeoUtils compares **eager, Dask and Multiprocessing** execution using the same deterministic inputs. Reprojection,
polygonization, rasterization and gridding additionally use equivalent **GDAL CLI commands** with matching
output grids, data types and storage options where possible. Every result identifies its operation, numerical method,
calculation engine, chunk strategy and execution mode separately. Engine comparisons use eager execution, while
execution-mode and chunk-strategy comparisons fix the other dimensions. Fixed-size Numba checks additionally exercise
each compiled gridding kernel in Dask and Multiprocessing workers. The Rasterio/GDAL calculation engine used inside
GeoUtils remains distinct from the external GDAL CLI reference.

Two reference graphics summarize the core results:

- **End-to-end time relative to GDAL** for each comparable operation
- **Peak RAM as raster size increases**, including the full process and its workers

:::{figure} https://glaciohack.github.io/geoutils/documentation/time_relative_to_gdal.svg
:alt: End-to-end GeoUtils execution-mode time relative to GDAL for four raster operations

End-to-end time on the largest raster size shared by every execution mode and the GDAL CLI. GDAL is the reference at one.
:::

:::{figure} https://glaciohack.github.io/geoutils/documentation/peak_ram_by_raster_size.svg
:alt: Peak process-tree RAM by raster size for GeoUtils execution modes and GDAL

Peak RAM for the benchmark process and all execution-mode workers as raster dimensions increase.
:::

These graphics show the latest complete CI benchmark and may be newer than this documentation version.
[Download the exact values and run metadata](https://glaciohack.github.io/geoutils/documentation/benchmark_snapshot.json).

## Test suite for scalable execution and large datasets

Every GeoUtils operation advertised as {ref}`chunked or lazy <scalability-support>` is tested
on all supported Python versions and operating systems for its respect of Dask laziness, deferred I/O, and loading behaviour,
while yielding **exactly** the same output as in-memory operations.

In addition, GeoUtils also includes large data tests running on the latest Python and Ubuntu,
verifying that these operations use less memory than loading the full raster would require, while yielding a correct result.
