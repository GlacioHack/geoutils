# GeoUtils benchmarks

This directory contains benchmarking tools to assess, record, compare and publish performance (compute time and memory usage) for GeoUtils main operations.

Mainly, it contains tools for running:
- Repeatable performance measurements (=benchmarking) that monitor and improve performance using [ASV](https://github.com/airspeed-velocity/asv),
  both to facilitate the check of a given functionality locally when developping, and to run a full benchmarking suite (~1h) that publishes
  detailed performance results to a GitHub page (updates for every PR merged into main),
- Large data tests for Pytest (pass or fail), which are less exhaustive but run faster (~10min) to quickly catch large regressions (runs every commit to a PR).

## Organization

- `workflows/` defines deterministic inputs (e.g. raster/point-cloud data), operations (e.g. ``reproject()``, ``grid()``),
  methods (e.g. ``resampling="linear"``), calculation engines (e.g., SciPy, Numba), chunk strategies (e.g., "dense" or
  "sparse" for grouped stats), and execution modes (eager, Dask, multiprocessing),
  to setup all possible computations that can be run by a given suite (ASV benchmark + large data tests),
- `asv_suite/operations.py` sets up ASV to measure individual operations at one fixed configuration,
- `asv_suite/parameter_sweeps.py` sets up ASV to measure operations across one-dimensional parameter ranges (e.g., raster input size, or method type),
- `asv_suite/render_results.py` renders the raw measurements into comparisons and graphics used by the GitHub pages and documentation,
- `gdal_comparison/` contains GDAL CLI equivalent operations for performance comparison,
- `pdal_comparison/` contains PDAL pipelines equivalent operations for performance comparison,
- `test_large_data.py` is a Pytest module to verify that every supported Dask/Multiprocessingoperation computes correctly without
  loading the complete raster into memory.

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

For example, `<benchmark-regex>` can be `EagerIdwNumbaGriddingRasterSize.time_operation` to benchmark ``grid(resampling="idw", engine="numba")``.

To compare the performance a new implementation with that of the `main` branch: commit current changes, then use:

```bash
asv continuous main HEAD -b 'EagerIdwNumbaGriddingRasterSize.time_operation'
```

To save the results and generate the HTML report locally:

```bash
asv run --show-stderr -E existing --bench 'EagerIdwNumbaGriddingRasterSize.time_operation'
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
table for eager, Dask and Multiprocessing end-to-end time normalized to the GDAL CLI on the same revision.

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
