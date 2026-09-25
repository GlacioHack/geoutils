# GeoUtils benchmarks

This directory contains benchmarking tools to assess, record, compare and publish performance (compute time and memory usage) for GeoUtils main operations.

Mainly, it contains tools for running:
- Repeatable performance measurements (=benchmarking) that monitor and improve performance using [ASV](https://github.com/airspeed-velocity/asv),
  both to facilitate the check of a given functionality locally when developping, and to run a full benchmarking suite (~1h) that publishes
  detailed performance results to a GitHub page (updates for every PR merged into main),
- Large data tests for Pytest (pass or fail), which are less exhaustive but run faster (~10min) to quickly catch large regressions (runs every commit to a PR).

## Organization

- `workflows/config.py` defines the shared benchmark configuration, parameter ranges and parameter sweeps,
- `workflows/fixtures.py` defines inputs used by benchmarks (e.g. synthetic raster/point-cloud data),
- `workflows/operations/` defines operations (e.g. ``reproject()``, ``grid()``), methods (e.g. ``resampling="linear"``),
  calculation engines (e.g., SciPy, Numba), chunk strategies (e.g., "dense" or "sparse" for grouped stats), and
  execution modes (in-memory, Dask, multiprocessing), beside the code that runs each operation,
- `workflows/runner.py` sets up the shared inputs, workers, profiling and result computation,
- `asv_suite/operations.py` sets up ASV to measure individual operations at one fixed configuration,
- `asv_suite/parameter_sweeps.py` sets up ASV to measure operations across one-dimensional parameter ranges (e.g., raster input size, or method type),
- `asv_suite/render_results.py` renders the raw measurements into comparisons and graphics used by the GitHub pages and documentation,
- `dask_comparison/` contains direct Dask equivalent operations for internal performance comparison,
- `flox_comparison/` contains direct Flox grouped statistics for performance comparison,
- `gdal_comparison/` contains GDAL CLI equivalent operations for performance comparison,
- `pdal_comparison/` contains PDAL pipelines equivalent operations for performance comparison,
- `test_large_data.py` is a Pytest module to verify that every supported Dask/Multiprocessing operation computes correctly without
  loading the complete raster into memory.

## Adding a benchmark

Benchmarks are defined in `workflows/operations/`, with one file per functionality (e.g., ``reprojection.py`` or ``filters.py``).
Each file describes which GeoUtils function to call (e.g., ``reproject()``), which cases to compare (e.g., Dask/Multiproc), which
parameter sweep to perform (e.g., varying raster input size) and finally if/how to compare to external references (e.g., GDAL/PDAL).

The benchmarks rely on five small objects:

- An `Operation` stores the functions that prepare and run one GeoUtils operation.
- A `BenchmarkCase` stores a fixed implementation, such as a method + engine + execution mode. It is returned by
  `execution_cases()`, `strategy_cases()` or `external_case()`, and each becomes one ASV benchmark.
- A `ParameterAxis` stores the name and values of one changing input, such as raster size or chunk size.
- A `Sweep` combines an axis, the function that applies its values and the cases to measure at every value.
- A `Comparison`, returned by `comparison()`, selects saved result series for one report plot. It can combine GeoUtils
  results with equivalent GDAL, PDAL or Flox results and does not run additional benchmarks.

To add a new benchmark, follow these steps:

1. Choose the file in `workflows/operations/` to add your new benchmark of a functionality, or create a new one.
2. Write the setup functions then register them with `Operation(...)`  (see examples in existing files).
3. Use `execution_cases()` or `strategy_cases()` to define the implementations to compare.
4. Choose or define a `ParameterAxis`, write the function that applies its values to `BenchmarkConfig`, then add
   `Sweep(axis, update, cases)` to the module's `SWEEPS`.
5. If you want to add an external implementation, add a `<name>_comparison/` directory, and attach it to the sweep with
   `external_case(...)`. Then, use `comparison(...)` in the functionality file to select the sweep  to compare with.

When running ASV, local outputs are generated under the gitignored `results/` directory:

```text
results/
├── asv/
│   ├── env/        # ASV environments
│   ├── results/    # Raw measurements
│   └── html/       # Combined website, starting at index.html
└── documentation/  # Optional local preview of the documentation graphics
```

## Performance benchmarks with ASV

### Working with our ASV benchmarks locally

Below a short summary on how to use our benchmarks locally, including both typical ASV commands and our custom routines.

To run a benchmark while developing:

```bash
asv run --quick --show-stderr -E existing --bench <benchmark-regex>
```

For example, `<benchmark-regex>` can be `grid_idw_numba_inmem__rastersize.time_operation` to benchmark ``grid(resampling="idw", engine="numba")``.
We use `_` between function parameters, and `__` before a varied input size.

To compare the performance a new implementation with that of the `main` branch: commit current changes, then use:

```bash
asv continuous main HEAD -b 'grid_idw_numba_inmem__rastersize.time_operation'
```

To save the results and generate the HTML report locally:

```bash
asv run --show-stderr -E existing --bench 'grid_idw_numba_inmem__rastersize.time_operation'
asv publish
python -m benchmarks.asv_suite.render_results
```

Then open `results/asv/html/index.html`.

To visualize the rendering of the GitHub page without running benchmarks, generate fake results and render them locally with:

```bash
python -m benchmarks.asv_suite.render_results --preview
asv preview --browser --html-dir benchmarks/results/asv/preview
```

Pass `--baseline-commit <commit>` to the renderer to add `comparisons/performance-change.md`, an before/after
table for in-memory, Dask and Multiprocessing end-to-end time normalized to the GDAL CLI on the same revision.

To generate the benchmarking figures used both on the benchmarking webpage + linked in the RTD documentation, run:

```bash
python -m benchmarks.asv_suite.render_results --doc-only --doc-dir benchmarks/results/documentation
```

### How our ASV benchmarks run in CI

Three workflows are related to ASV benchmark in our continuous integration.

1. For every PR commit, `benchmark-asv-check` runs a quick check (~10min) of benchmark setups (it uses the `GEOUTILS_ASV_PR_CHECK=1`
environment variable to run quick check on reduced parameters, and otherwise relies on ``asv check``).

2. For every PR merge into `main`, or on weekly schedule, `benchmark-asv` runs the full suite (1h+) and records
the performance results to an `asv-results` branch on the GeoUtils repository (relying on ``asv run``).

3. After a successful run of the previous `benchmark-asv`, `benchmark-publish` automatically rebuilds the
saved history from `asv-results`, re-renders the graphics, and deploys the benchmarking webpage to GitHub Pages.

Note that the last two workflows can be triggered manually. Additionally, the user documentation on ReadTheDocs contains direct
links to the latest graphics published on the Benchmarking page (which are therefore always updated to the latest benchmark build,
on all documentation versions).


## Large data tests

Normal Pytest tests (that run across all OSs and Python versions) skip large data tests by default.
In the CI, they run once on Ubuntu with Python 3.12 for every PR commit.

Locally, run the complete suite with:

```bash
python -m pytest --large-data -m large_data -ra
```

To save time while developing, select one specific test with `-k`  and/or add `--lf` to repeat only failed cases.
Other practical instructions and environment variables to set are documented at the top of `test_large_data.py`.
