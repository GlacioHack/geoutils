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
(profiling)=
# Profiling

GeoUtils has a **built-in profiling tool to measure execution time and memory usage** of a function with your own data and hardware.

This profiling tool supports the **systematic benchmarking** of the package presented in {ref}`benchmarking-performance`.

```{note}
Profiling functionalities rely on [psutil](https://psutil.readthedocs.io/en/latest/) and [plotly](https://plotly.com/) as optional dependencies. You can install them manually or with ``pip install geoutils[prof]``
```

## Profiling a single function call

For a one-off measurement, {func}`~geoutils.profiler.profile_call` runs a function and returns both its output and normalized metrics for
runtime and memory. The current Python process is measured with psutil. If a distributed Dask client is active,
GeoUtils also uses Dask's {class}`~distributed.diagnostics.MemorySampler` to report worker process memory and spilled memory.

```{code-cell} python
import geoutils as gu
from geoutils.profiler import profile_call

# Load the example before starting the measurement
raster = gu.Raster(gu.examples.get_path("exploradores_aster_dem"))

# Measure a real reprojection
reprojected, metrics = profile_call(raster.reproject, crs=4326)
{"runtime_s": metrics.runtime_s, "peak_client_mem_mb": metrics.peak_client_mem_mb}
```

The returned {class}`~geoutils.profiler.ProfileMetrics` gives direct access to the runtime, peak values and raw memory
samples. Calling {meth}`metrics.plot() <geoutils.profiler.ProfileMetrics.plot>` returns an interactive figure of all available traces. Pass `include_children=True`
to also measure aggregate process-tree memory and the largest child process, for example for Multiprocessing or subprocess
workflows. Pass `profile_memory=False` to measure runtime without starting memory samplers.

## Profiling a workflow

For profiling a workflow, decorate with {obj}`@profile <geoutils.profiler.profile>` to mark the calls to include, and use {class}`~geoutils.profiler.Profiler` to collect them in
order and display their memory/time usage.

### Configuration and parameters

Collection is disabled by default and is enabled by {meth}`Profiler.enable() <geoutils.profiler.Profiler.enable>` when at least one of these parameters is `True`:

| Name              | Description                                      | Type | Default value | Required |
|-------------------|--------------------------------------------------|------|---------------|----------|
| **save_graphs**   | Save the default graphs generated                | bool | False         | No       |
| **save_raw_data** | Save the raw data on calls as a `.pickle` file   | bool | False         | No       |

```{code-cell} python
from IPython.display import HTML, display
from geoutils.profiler import Profiler, profile

Profiler.enable(save_graphs=True, save_raw_data=True)
```

Every decorated function called after this is recorded by the profiler.

### The profiled functions

GeoUtils profiles the shared implementations of its core numerical operations, including {meth}`~RasterBase.reproject`, {meth}`~RasterBase.crop`,
{meth}`~RasterBase.polygonize`, {meth}`~VectorBase.rasterize`, {meth}`~PointCloudBase.grid`, {meth}`~RasterBase.stats`, {meth}`~RasterBase.subsample`, {meth}`~RasterBase.filter`, {meth}`~RasterBase.interp_at_points`, {meth}`~RasterBase.sieve` and {meth}`~RasterBase.fill_nodata`.
Object methods and Pandas or Xarray accessors therefore record the same underlying operation where supported, without
also recording their lightweight wrappers. Memory is sampled at the interval configured by each decorator, which
defaults to 0.005 seconds.

#### Modifying the profiled functions

To add another function to the summary, decorate it with {obj}`@profile <geoutils.profiler.profile>` and provide a descriptive name. Set `memprof=True`
to track memory over time, and adjust `interval` if the function is too fast or slow for the default sampling interval.

```{code-cell} python
@profile("raster workflow", memprof=True, interval=0.01)
def raster_workflow(source_raster):
    left, bottom, right, top = source_raster.bbox
    cropped = source_raster.crop((left, bottom, (left + right) / 2, top))
    reprojected = cropped.reproject(crs=4326)
    filtered = reprojected.filter("mean", window=3)
    return filtered.stats(["mean", "std", "nmad"])

# The workflow and its four numerical operations are added to the collection
statistics = raster_workflow(raster)
```

| Name         | Description                                      | Type  | Default value | Required |
|--------------|--------------------------------------------------|-------|---------------|----------|
| **name**     | Name of the function in the report               | str   |               | Yes      |
| **interval** | Memory sampling interval (seconds)               | float | 0.005         | No       |
| **memprof**  | Whether to profile the memory consumption        | bool  | False         | No       |
| **collect**  | Whether calls to this function are collected     | bool  | True          | No       |

A decorated function that returns a lazy Dask collection records worker activity only when computation occurs inside
the call. Otherwise, profile the terminal computation explicitly with `profile_call(lambda: collection.compute())`.

### Output graphs

The collected hierarchy and memory traces are available directly as interactive Plotly figures through
{meth}`Profiler.plot_time_summary() <geoutils.profiler.Profiler.plot_time_summary>` and
{meth}`Profiler.plot_trace_for_call() <geoutils.profiler.Profiler.plot_trace_for_call>`:

```{code-cell} python
workflow_call = Profiler.get_profiling_info("raster workflow").iloc[0]
time_figure = Profiler.plot_time_summary()
workflow_memory_figure = Profiler.plot_trace_for_call(workflow_call["uuid_function"], "memory")

display(HTML(time_figure.to_html(full_html=False, include_plotlyjs="cdn")))
display(HTML(workflow_memory_figure.to_html(full_html=False, include_plotlyjs="cdn")))
```

The plotting methods optionally accept a path to save their figure. With `save_graphs=True`,
{meth}`Profiler.generate_summary(output) <geoutils.profiler.Profiler.generate_summary>` saves two kinds of HTML graph:

- `time_graph.html`, an icicle graph showing the time spent in each decorated call
- one `memory_[function].html` graph per memory-profiled call, showing the client process and any available Dask-worker
  memory traces

### Saved profiling data

{meth}`Profiler.generate_summary(output) <geoutils.profiler.Profiler.generate_summary>` saves profiling information in the given directory, or in `output_profiling` when
no directory is specified. With `save_raw_data=True`, it writes `raw_data.pickle`, which contains a
{class}`~pandas.DataFrame` with this structure:

| Name              | Description                                                                                                                                       |
|-------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| **level**         | Depth of the function call in the profiling stack                                                                                                 |
| **uuid_function** | Unique universal identifier (UUID) of the function call                                                                                           |
| **name**          | Descriptive name given to the function call                                                                                                       |
| **uuid_parent**   | UUID of the parent call that was running when this call was made                                                                                   |
| **time**          | Time in seconds taken to execute the function                                                                                                     |
| **call_time**     | Timestamp in seconds at which the call was made                                                                                                   |
| **memory**        | `None` or a list of `(timestamp, memory)` tuples representing client memory in megabytes during the call                                           |
| **metrics**       | Complete {class}`~geoutils.profiler.ProfileMetrics` returned by {func}`~geoutils.profiler.profile_call`, including client, child-process and Dask measurements            |

If no decorated function has been recorded, {meth}`~geoutils.profiler.Profiler.generate_summary` produces no output.
