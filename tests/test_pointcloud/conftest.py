"""Shared fixtures for point cloud tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def copc_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    Write a COPC file with two levels of details and an integer column for tests.

    We have to use PDAL as LasPy does not yet support COPC writing.

    The pipeline uses the native writer documented at https://pdal.io/en/stable/stages/writers.copc.html.
    """

    # Check optional dependencies before loading PDAL COPC writer
    pytest.importorskip("laspy")
    pytest.importorskip("lazrs")
    pdal = pytest.importorskip("pdal")

    # We create enough points for multiple levels of details
    point_numbers = np.arange(100_003)
    grid_width = 317
    points = np.zeros(point_numbers.size, dtype=[("X", "f8"), ("Y", "f8"), ("Z", "f8"), ("Intensity", "u2")])
    points["X"] = point_numbers % grid_width
    points["Y"] = point_numbers // grid_width
    points["Z"] = points["X"] * 0.1 + points["Y"] * 0.2

    # We change dtype for an auxiliary attribute, to check reading in tests
    points["Intensity"] = point_numbers.astype(np.uint16)

    # Write the file
    filename = tmp_path_factory.mktemp("copc") / "points.copc.laz"
    stages = [{"type": "writers.copc", "filename": str(filename), "a_srs": "EPSG:32633"}]
    pipeline = pdal.Pipeline(json.dumps(stages), arrays=[points])
    assert pipeline.execute() == len(points)
    return filename
