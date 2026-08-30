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

GeoUtils has a **built-in profiling tool to measure time and memory** used by a function on your own data and hardware.
The same measurements support the controlled comparisons presented in {ref}`benchmarking-performance`.

```{note}
The profiling functionalities rely on [psutil](https://psutil.readthedocs.io/en/latest/) and [plotly](https://plotly.com/) as optional dependencies. You can install them manually or with ``pip install geoutils[opt]``
```

## Profiling a single function call

For a one-off measurement, `profile_call()` runs a function and returns both its output and normalized metrics for
runtime and memory. The current Python process is measured with psutil. If a distributed Dask client is active,
GeoUtils also uses Dask's `MemorySampler` to report worker process memory and spilled memory.

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
samples. Calling `metrics.plot()` returns an interactive figure of all available traces. Pass `include_children=True`
to also measure aggregate process-tree memory and the largest child process, for example for Multiprocessing or subprocess
workflows. Pass `profile_memory=False` to measure runtime without starting memory samplers.

## Profiling a workflow

For profiling a workflow, decorate with `@profile` to mark the calls to include, and use `Profiler` to collect them in
order and display their memory/time usage.

### Configuration and parameters

Collection is disabled by default and is enabled when at least one of these parameters is `True`:

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

GeoUtils profiles the shared implementations of its core numerical operations, including `reproject`, `crop`,
`polygonize`, `rasterize`, `grid`, `get_stats`, `subsample`, `filter`, `interp_points`, `sieve` and `fill_nodata`.
Object methods and Pandas or Xarray accessors therefore record the same underlying operation where supported, without
also recording their lightweight wrappers. Memory is sampled at the interval configured by each decorator, which
defaults to 0.005 seconds.

#### Modifying the profiled functions

To add another function to the summary, decorate it with `@profile` and provide a descriptive name. Set `memprof=True`
to track memory over time, and adjust `interval` if the function is too fast or slow for the default sampling interval.

```{code-cell} python
@profile("raster workflow", memprof=True, interval=0.01)
def raster_workflow(source_raster):
    left, bottom, right, top = source_raster.bounds
    cropped = source_raster.crop((left, bottom, (left + right) / 2, top))
    reprojected = cropped.reproject(crs=4326)
    filtered = reprojected.filter("mean", size=3)
    return filtered.get_stats(["mean", "std", "nmad"])

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

The collected hierarchy and memory traces are available directly as interactive Plotly figures:

```{code-cell} python
workflow_call = Profiler.get_profiling_info("raster workflow").iloc[0]
time_figure = Profiler.plot_time_summary()
workflow_memory_figure = Profiler.plot_trace_for_call(workflow_call["uuid_function"], "memory")

display(HTML(time_figure.to_html(full_html=False, include_plotlyjs="cdn")))
display(HTML(workflow_memory_figure.to_html(full_html=False, include_plotlyjs="cdn")))
```

The plotting methods optionally accept a path to save their figure. With `save_graphs=True`,
`Profiler.generate_summary(output)` saves two kinds of HTML graph:

- `time_graph.html`, an icicle graph showing the time spent in each decorated call
- one `memory_[function].html` graph per memory-profiled call, showing the client process and any available Dask-worker
  memory traces

### Saved profiling data

`Profiler.generate_summary(output)` saves profiling information in the given directory, or in `output_profiling` when
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
| **metrics**       | Complete {class}`~geoutils.profiler.ProfileMetrics` returned by `profile_call()`, including client, child-process and Dask measurements            |

If no decorated function has been recorded, `generate_summary()` produces no output.
