(benchmarking-performance)=
# Performance

GeoUtils benchmarks its functionalities to measure **RAM and wall time**, check **lazy and out-of-core behaviour** with
**Dask and Multiprocessing**, track improvements or regressions over time, and compare the performance of different
backends and the [**GDAL CLI**](https://gdal.org/en/stable/programs/index.html).

Measurements combine [**psutil**](https://psutil.readthedocs.io/) process measurements with **Dask** worker and spilled-memory diagnostics, as described in {ref}`profiling`.

## Monitoring improvements or regressions over time

GeoUtils uses [**Airspeed Velocity (ASV)**](https://asv.readthedocs.io/) to record fixed performance measurements for each commit. This
makes changes in time or memory visible while keeping comparisons on the same machine and software environment.

The [**GeoUtils benchmark webpage**](https://glaciohack.github.io/geoutils/) provides the performance changes with interactive commit history.

## Backend and GDAL comparisons

GeoUtils compares **eager, Dask and Multiprocessing** execution using the same deterministic inputs. Reprojection,
polygonization, rasterization and gridding additionally use equivalent **GDAL file-to-file commands** with matching
output grids, data types and storage options where possible.

Two reference graphics summarize the core results:

- **End-to-end time relative to GDAL** for each comparable operation
- **Peak RAM as raster size increases**, including the full process and its workers

```{include} imgs/benchmarking/summary.md.inc
```

## Large data tests

Every GeoUtils operation advertised as {ref}`chunked or lazy <scalability-support>` is also tested with an input whose
uncompressed size exceeds the memory allowance. It must compute with a correct result using less additional worker
memory than loading the full raster would require.
