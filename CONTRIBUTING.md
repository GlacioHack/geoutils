# How to contribute

## Overview: making a contribution

For more details, see the rest of this document.

1. Fork _GlacioHack/geoutils_ and clone your fork repository locally.
2. Set up the development environment (section below).
3. Create a branch for the new feature or bug fix.
4. Make your changes, and add or modify related tests in _tests/_.
5. Commit, making sure to run `pre-commit` separately if not installed as git hook.
6. Push to your fork.
7. Open a pull request from GitHub to discuss and eventually merge.

## Development environment

GeoUtils currently supports only Python versions of 3.10 to 3.14, see `environment.yml` for detailed dependencies.

### Setup

Clone the git repo and create a `mamba` environment (see how to install `mamba` in the [mamba documentation](https://mamba.readthedocs.io/en/latest/)):

```bash
git clone https://github.com/GlacioHack/geoutils.git
cd geoutils
mamba env create -f dev-environment.yml  # Add '-n custom_name' if you want.
mamba activate geoutils-dev  # Or any other name specified above
```

### Tests

At least one test per feature (in the associated `tests/test_*.py` file) should be included in the PR, using `pytest` (see existing tests for examples).

To run the entire test suite, run `pytest` in the current directory:

```bash
pytest
```

### Performance changes

Use a targeted quick run while developing benchmarked code. Quick measurements check that the workflow executes, but
are not reliable performance results and are not saved:

```bash
asv run --quick --show-stderr -E existing --bench implementation_comparisons
```

For a repeatable local result, commit the code, omit `--quick`, then run `asv publish` followed by
`python -m benchmarks.asv_suite.render_results`. Open `benchmarks/results/asv/html/index.html` for both the ASV history
and the implementation-comparison plots. Compare timings only between results collected on the same machine and
environment.

Reference graphics in the user documentation are updated separately from saved ASV history. Maintainers can trigger
the `benchmark-documentation` workflow to render the newest complete CI result and open a reviewable documentation PR.

Operation time excludes backend initialization, end-to-end time includes it but excludes input generation, and peak
RAM covers the benchmark process and all child processes. Memory is measured separately so its sampling does not alter
the timing result.

Normal Pytest skips the large data suite. Pull-request CI runs it when scalable backend code or its workflows change.
When investigating one case locally, select it explicitly and use `--lf` on later runs to repeat only failures:

```bash
pytest --large-data -m large_data -k dask-reproject -ra
pytest --large-data -m large_data --lf -ra
```

The full suite is intentionally long and memory-intensive. See the [benchmark directory guide](benchmarks/README.md)
for its configuration and the complete ASV commands. Further maintainer guidance is kept in the project Wiki.

### Formatting and linting

Install and run `pre-commit` (see [pre-commit documentation](https://pre-commit.com/)), which will use `.pre-commit-config.yaml` to verify spelling errors,
import sorting, type checking, formatting and linting.

You can then run pre-commit manually:

```bash
pre-commit run --all-files
```

Optionally, `pre-commit` can be installed as a git hook to ensure checks have to pass before committing.

## Rights

The license (see LICENSE) applies to all contributions.
