(benchmarking-performance)=
# Performance

GeoUtils benchmarks its functionalities to measure **RAM and wall time**, check **lazy and out-of-core behaviour** with
**Dask and Multiprocessing**, track improvements or regressions over time, and compare the performance across different
backends and against the [**GDAL CLI**](https://gdal.org/en/stable/programs/index.html).

The benchmarks use GeoUtils' profiling tool described in {ref}`profiling`, which relies on [**psutil**](https://psutil.readthedocs.io/) and [**Dask's worker memory diagnostics**](https://distributed.dask.org/en/stable/worker-memory.html).

## Performance and its improvements or regressions over time

The [**GeoUtils benchmark webpage**](https://glaciohack.github.io/geoutils/) provides performance of various functionalities and their changes with commit history.

For this, GeoUtils relies on [**Airspeed Velocity (ASV)**](https://asv.readthedocs.io/) to record fixed performance measurements for each commit. This
makes changes in time or memory visible while keeping comparisons on the same machine and software environment.

## Comparison across backends and against GDAL CLI

GeoUtils compares **eager, Dask and Multiprocessing** execution using the same deterministic inputs. Reprojection,
polygonization, rasterization and gridding additionally use equivalent **GDAL CLI commands** with matching
output grids, data types and storage options where possible.

Two reference graphics summarize the core results:

- **End-to-end time relative to GDAL** for each comparable operation
- **Peak RAM as raster size increases**, including the full process and its workers

```{include} imgs/benchmarking/summary.md.inc
```

## Test suite for scalable execution and large datasets

Every GeoUtils operation advertised as {ref}`chunked or lazy <scalability-support>` is tested
on all supported Python versions and operating systems for its respect of Dask laziness, deferred I/O, and loading behaviour;
which can all be verified on small test data.

In addition, GeoUtils also includes large data tests running on the latest Python and Ubuntu,
verifying that these operations yield a correct result using less memory than loading the full raster would require.
