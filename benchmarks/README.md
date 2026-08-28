# GeoUtils benchmarks

This directory contains repeatable performance measurements and pass/fail large data tests.

## Organization

- `workflows/` defines deterministic inputs, supported operation/backend pairs and final result computation shared by
  every suite (ASV benchmark + large data tests),
- `asv_suite/operations.py` measures each operation with one fixed configuration of input parameters,
- `asv_suite/comparisons.py` measures how time and RAM vary across specific input parameters (e.g. raster size) for eager,
  Dask, Multiprocessing and, when available, GDAL.
- `asv_suite/render_results.py` adds some custom routines to render the ASV raw measurements into the benchmark website page (most is already done by ASV there) and GDAL comparison plots for the documentation,
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
└── documentation/  # Optional local preview of the reviewed graphics
```

## Performance benchmarks

To run a benchmark while developing:

```bash
asv run --quick --show-stderr -E existing --bench <benchmark-regex>
```

To save the results and generate the HTML report locally, omit `--quick`, then run `asv publish` followed by
`python -m benchmarks.asv_suite.render_results`. Then open `results/asv/html/index.html` for the native ASV
history and the comparison plots. In CI, all of this is automated and uploads to GitHub Pages.

To update the benchmarking graphics in the documentation, trigger the manual `benchmark-link-docs` GitHub workflow,
which pulls the latest published ASV results, then review and merge the pull request it opens.

For an optional local preview without changing version-controlled files, run:

```bash
python -m benchmarks.asv_suite.render_results --doc-only --doc-dir benchmarks/results/documentation
```

## Large data tests

Normal Pytest skips these intentionally expensive checks, while pull-request CI always runs them once on Ubuntu with
Python 3.12. Run the complete suite locally with:

```bash
pytest --large-data -m large_data -ra
```

Select one parameter with `-k` while developing and add `--lf` to repeat only failed cases. The practical instructions
and environment variables are documented at the top of `test_large_data.py`.
