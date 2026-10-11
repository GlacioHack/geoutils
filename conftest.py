"""Repository-wide pytest config, to reach benchmarks/ as well as tests/, and skip large data tests by default."""

from __future__ import annotations

from doctest import ELLIPSIS, NORMALIZE_WHITESPACE
from importlib.util import find_spec
from typing import Any

import pytest
from sybil import Sybil
from sybil.parsers.rest import DocTestParser, PythonCodeBlockParser


def _setup_docstring_examples(namespace: dict[str, Any]) -> None:
    """Provide small raster and point inputs for the spatial class examples."""

    import numpy as np
    from rasterio.transform import from_origin

    import geoutils as gu

    # Plane values for interpolation and a single output cell for grid examples
    values = np.add.outer(np.arange(5.0), np.arange(5.0))
    raster = gu.Raster.from_array(values, from_origin(0, 5, 1, 1), crs=32631)
    reference_raster = gu.Raster.from_array(np.zeros((1, 1)), from_origin(0, 5, 5, 5), crs=32631)

    # Four points around the output cell
    point_cloud = gu.PointCloud.from_xyz([0.5, 4.5, 0.5, 4.5], [0.5, 0.5, 4.5, 4.5], [4, 8, 0, 4], crs=32631)
    namespace.update(raster=raster, reference_raster=reference_raster, point_cloud=point_cloud, x=2.5, y=2.5)


_docstring_example_files = ["geoutils/operators/interpolator.py", "geoutils/operators/reducer.py"]
if find_spec("skgstat") is not None:
    _docstring_example_files.append("geoutils/uncertainty/error_structure.py")

# Collect existing prompt examples and the selected class code blocks
_doctest_examples = Sybil(parsers=[DocTestParser(optionflags=ELLIPSIS | NORMALIZE_WHITESPACE)], patterns=["*.py"])
_code_block_examples = Sybil(
    parsers=[PythonCodeBlockParser()], patterns=_docstring_example_files, setup=_setup_docstring_examples
)
pytest_collect_file = (_doctest_examples + _code_block_examples).pytest()


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom GeoUtils test options."""

    parser.addoption(
        "--large-data",
        action="store_true",
        default=False,
        help="Run large data tests.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip large data tests unless explicitly requested."""

    if config.getoption("--large-data"):
        return

    skip_large_data = pytest.mark.skip(reason="Large data test; use --large-data to run.")
    for item in items:
        if "large_data" in item.keywords:
            item.add_marker(skip_large_data)
