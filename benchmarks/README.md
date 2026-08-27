# GeoUtils benchmarks

This directory contains repeatable performance measurements and pass/fail large data tests.

## Organization

- `workflows/` defines deterministic inputs, supported operation/backend pairs and final result computation shared by
  every suite
- `asv_suite/operations.py` measures each registered operation at one fixed configuration
- `asv_suite/implementation_comparisons.py` measures how time and RAM vary with one input parameter across eager,
  Dask, Multiprocessing and, where available, GDAL. Gridding covers nearest, linear, inverse-distance and circular-mean
  methods without repeating equivalent circular statistics, plus the SciPy/Numba nearest crossover with point count
- `asv_suite/render_results.py` turns saved ASV measurements into the combined website and reviewed documentation
  graphics
- `gdal_comparison/` builds and runs equivalent GDAL file-to-file operations
- `test_large_data.py` verifies that every supported Dask and Multiprocessing operation computes correctly without
  loading the complete raster into memory

All local outputs are stored under the gitignored `results/` directory:

```text
results/
├── asv/
│   ├── env/       # ASV environments
│   ├── results/   # Raw measurements
│   └── html/      # Combined website, starting at index.html
└── documentation/ # Optional local preview of the reviewed graphics
```

The selected documentation snapshot is copied separately to `doc/source/imgs/benchmarking/` because those files are
reviewed and version-controlled.

## Performance benchmarks

Run one affected benchmark while developing:

```bash
asv run --quick --show-stderr -E existing --bench <benchmark-regex>
```

For a saved result, omit `--quick`, then run `asv publish` followed by
`python -m benchmarks.asv_suite.render_results`. Open `results/asv/html/index.html` for the native ASV history and the
implementation-comparison plots.

To preview only the documentation graphics without changing version-controlled files, run:

```bash
python -m benchmarks.asv_suite.render_results --documentation-only \
    --documentation-directory benchmarks/results/documentation
```

Maintainers can trigger the manual `benchmark-documentation` GitHub workflow to render the newest complete CI result,
upload it as an artifact and open a pull request updating the user documentation.

## Large data tests

Normal Pytest skips these intentionally expensive checks. Run the complete suite with:

```bash
pytest --large-data -m large_data -ra
```

Select one parameter with `-k` while developing and add `--lf` to repeat only failed cases. The practical instructions
and environment variables are documented at the top of `test_large_data.py`.

Raster merging remains excluded until it provides equivalent file-to-file Dask or Multiprocessing behavior. See
[CONTRIBUTING.md](../CONTRIBUTING.md) for contributor expectations and the
[benchmarking reference](../doc/source/benchmarking_index.md) for user-facing interpretation.
