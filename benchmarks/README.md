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

To run a benchmark while developing:

```bash
asv run --quick --show-stderr -E existing --bench <benchmark-regex>
```

For example, `<benchmark-regex>` can be `EagerIdwNumbaGriddingRasterSize.time_operation`.

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

Then open `results/asv/html/index.html` for the native ASV history and comparison plots.

Pass `--baseline-commit <commit>` to the renderer to add `comparisons/performance-change.md`, a compact before/after
table for eager, Dask and Multiprocessing end-to-end time normalized to the GDAL CLI on the same revision.

In CI, `benchmark-asv-check` verifies changed benchmarks on every pull request (using `GEOUTILS_ASV_PR_CHECK=1` to use
reduced parameters). The weekly or manually triggered `benchmark-asv` workflow records measurements on new `main`
commits and stores their raw history on
the `asv-results` branch of this repository. After a successful run, `benchmark-publish` automatically
rebuilds the latest saved history and deploys the website and documentation graphics to GitHub Pages.
Trigger it manually only to rebuild these outputs without new measurements.
The user documentation links to the latest complete graphics published there.

The benchmark dependencies include Pytest because ASV imports every Python module under `benchmarks/` during
discovery, including the large data test, without executing its tests.

For a local preview of the documentation graphics, run:

```bash
python -m benchmarks.asv_suite.render_results --doc-only --doc-dir benchmarks/results/documentation
```

## Large data tests

Normal Pytest skips these intentionally expensive checks, while pull-request CI always runs them once on Ubuntu with
Python 3.12. Run the complete suite locally with:

```bash
python -m pytest --large-data -m large_data -ra
```

Select one parameter with `-k` while developing and add `--lf` to repeat only failed cases. The practical instructions
and environment variables are documented at the top of `test_large_data.py`.
