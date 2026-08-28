# GeoUtils benchmarks

This directory contains repeatable performance measurements and pass/fail large data tests.

## Organization

- `workflows/` defines deterministic inputs, combinations of operation/backends and result computation shared by
  every suite (ASV benchmark + large data tests),
- `asv_suite/operations.py` measures each operation with one fixed configuration of input parameters,
- `asv_suite/comparisons.py` measures certain operations across specific input parameters (e.g. raster size) for eager,
  Dask, Multiprocessing and, when available, GDAL.
- `asv_suite/render_results.py` renders the ASV raw measurements into implementation comparisons and the two concise
  graphics used by the documentation,
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

To save the results and generate the HTML report locally, omit `--quick`, then run `asv publish` followed by
`python -m benchmarks.asv_suite.render_results`. Then open `results/asv/html/index.html` for the native ASV
history and the comparison plots.

In CI, `benchmark-asv-check` verifies changed benchmarks on every pull request. The weekly or manually triggered
`benchmark-asv` workflow records measurements on new `main` commits, and stores their raw history on
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
