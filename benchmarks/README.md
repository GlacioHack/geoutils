# GeoUtils benchmarks

This directory contains repeatable performance measurements and pass/fail large data tests.

## Organization

- `workflows/` defines deterministic inputs, operation methods, calculation engines, chunk strategies, execution modes
  and result computation shared by every suite (ASV benchmark + large data tests),
- `asv_suite/operations.py` measures operations without a dedicated scaling comparison at one fixed configuration,
- `asv_suite/comparisons.py` defines one-axis comparisons and generates their valid ASV cases and classes, with fixed
  Numba worker checks and the GDAL CLI kept as a separate external reference,
- `asv_suite/render_results.py` renders the raw measurements into method, engine, strategy and execution-mode
  comparisons and the two concise graphics used by the documentation,
- `gdal_comparison/` contains the GDAL CLI equivalent operations for performance comparison,
- `test_large_data.py` verifies that every supported Dask and Multiprocessing operation computes correctly without
  loading the complete raster into memory.

All local outputs are stored under the gitignored `results/` directory:

```text
results/
├── asv/
│   ├── env/        # ASV environments
│   ├── results/    # Raw measurements
│   └── html/       # Combined website, starting at index.html
└── documentation/  # Optional local preview of the documentation graphics
```

## Performance benchmarks

### Run benchmarks quickly while developing

To run a benchmark while developing:

```bash
asv run --quick --show-stderr -E existing --bench <benchmark-regex>
```

For example, `<benchmark-regex>` can be `EagerIdwNumbaGriddingRasterSize.time_operation`.

Set `GEOUTILS_ASV_PR_CHECK=1` to use the reduced parameter ranges from the pull-request checks:

```bash
GEOUTILS_ASV_PR_CHECK=1 asv run --quick --show-stderr -E existing --bench <benchmark-regex>
```

Omit `GEOUTILS_ASV_PR_CHECK=1` to use the complete parameter ranges.

To compare a new implementation with the `main` branch: commit current changes, then use:

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

To modify the rendering of the custom webpages without running benchmarks, generate fake results and render them locally with:

```bash
python -m benchmarks.asv_suite.render_results --preview
asv preview --browser --html-dir benchmarks/results/asv/preview
```

Pass `--baseline-commit <commit>` to the renderer to add `comparisons/performance-change.md`, a compact before/after
table for eager, Dask and Multiprocessing end-to-end time normalized to the GDAL CLI on the same revision.

For a local preview of the documentation graphics, run:

```bash
python -m benchmarks.asv_suite.render_results --doc-only --doc-dir benchmarks/results/documentation
```

### Benchmark structure and parameters

Benchmarks use two structures. `OperationBenchmarks` measures operations without a dedicated scaling comparison at one
fixed configuration. Its `case` parameter combines the execution mode and operation, such as `dask-reproject`.
Comparison benchmarks vary one input at a time. Their class names identify the fixed execution mode, method and
calculation engine. For example, `EagerIdwNumbaGriddingRasterSize` varies `raster_size` while keeping eager execution,
IDW and Numba fixed.

#### Input parameters

- `raster_size` is the number of pixels along each side of a square raster.
- `chunk_size` is the number of pixels along each side of a square chunk.
- `interpolated_points` is the number of raster locations queried by interpolation.
- `subsample_size` is the requested number of valid observations.
- `points_per_axis` defines a square point grid, so the total point count is its square.
- `groups_per_axis` defines groups along both raster axes, so the total group count is its square.

Some comparisons also vary input layout. For grouped statistics, local groups occupy contiguous areas, while
interleaved groups are spread across chunks. The benchmarks use two value arrays with different missing cells.

#### Implementation options

Comparison series vary one implementation choice while keeping the others fixed:

- `method` selects the algorithm used by an operation. For grouped statistics, `moments` computes count, mean, standard
  deviation and extrema, while `robust` computes exact median and NMAD.
- `calculation_engine` selects the numerical library that performs the calculation, such as SciPy or Numba.
- `strategy` selects how a chunked operation coordinates or combines results between chunks. Grouped statistics use
  `dense` to store every declared group in each chunk, `sparse` to store only groups present in a chunk, and `groupwise`
  to gather complete groups for exact statistics. `auto` selects `groupwise` for exact statistics and switches from
  `dense` to `sparse` above 4096 groups for mergeable statistics.
- `execution_mode` selects eager, Dask or Multiprocessing execution.

The GDAL CLI remains a separate external reference. Quick runs validate execution and reporting rather than repeatable
performance differences. The user statistics guide explains the memory limits of the grouped-statistics strategies.

### Continuous integration and published reports

In CI, `benchmark-asv-check` verifies changed benchmarks on every pull request (using `GEOUTILS_ASV_PR_CHECK=1` to use
reduced parameters). The weekly or manually triggered `benchmark-asv` workflow records measurements on new `main`
commits and stores their raw history on
the `asv-results` branch of this repository. After a successful run, `benchmark-publish` automatically
rebuilds the latest saved history and deploys the website and documentation graphics to GitHub Pages.
Trigger it manually only to rebuild these outputs without new measurements.
The user documentation links to the latest complete graphics published there.

The options and scaling pages include every comparison, including grouped statistics. The compact GDAL documentation
graphics remain focused on operations with GDAL equivalents.

The benchmark dependencies include Pytest because ASV imports every Python module under `benchmarks/` during
discovery, including the large data test, without executing its tests.

## Large data tests

### Run large data tests

Normal Pytest skips these intentionally expensive checks, while pull-request CI always runs them once on Ubuntu with
Python 3.12. Run the complete suite locally with:

```bash
python -m pytest --large-data -m large_data -ra
```

Select one parameter with `-k` while developing and add `--lf` to repeat only failed cases. The practical instructions
and environment variables are documented at the top of `test_large_data.py`.

Grouped statistics have Dask memory contracts for dense and sparse summaries and exact localized medians/NMAD,
each at two chunk sizes. Multiprocessing grouping currently loads source arrays in the client before worker tiling,
so it is benchmarked but does not claim the same larger-than-memory contract. Run only grouped memory checks with
`python -m pytest --large-data -m large_data -k grouped_stats -ra`.
