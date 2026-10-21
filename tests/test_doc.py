"""Functions to test the documentation."""

import re
from importlib.util import find_spec
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import rasterio as rio
import xarray as xr
from pyproj import CRS
from rasterio.transform import from_origin

import geoutils as gu
from geoutils._misc import import_optional


@pytest.mark.skipif(
    any(find_spec(name) is None for name in ("dask", "laspy", "lazrs", "skgstat", "gstools")),
    reason="The README quick start requires Dask, LasPy with LAZ support, SciKit-GStat and GSTools.",
)
class TestReadmeChunked:
    """Test module for the README variogram, kriging and statistics workflow on chunked files."""

    @pytest.mark.parametrize("keep_attrs", [False, True])
    def test_readme_quick_start(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keep_attrs: bool) -> None:
        """Checks that the README code preserves lazy point results and recovers a known elevation bias."""

        # 1/ Write a smooth DEM split into 512 + 8 cells along both axes by the README chunk sizes
        monkeypatch.chdir(tmp_path)
        rows, columns = np.mgrid[:520, :520]
        elevations = (100 + 10 * np.sin(columns / 60) + 6 * np.cos(rows / 90)).astype("float32")
        transform = from_origin(500_000, 5_200, 10, 10)
        with rio.open(
            "dem.tif",
            "w",
            driver="GTiff",
            width=520,
            height=520,
            count=1,
            dtype="float32",
            crs=32631,
            transform=transform,
            nodata=-9999,
        ) as dataset:
            dataset.write(elevations, 1)

        # 2/ Write 10,003 LiDAR points with a 2 m bias and 0.2 m noise at known pixel centers
        # Repeat 40 locations, including both sides of raster chunk boundaries, to limit kriging work
        laspy = import_optional("laspy")
        point_count = 10_003
        locations = np.arange(point_count) % 40
        pixel_indices = np.concatenate((np.linspace(4, 510, 36, dtype=int), [511, 512, 513, 515]))
        point_rows = pixel_indices[locations]
        point_columns = pixel_indices[::-1][locations]
        noise = np.random.default_rng(42).normal(0, 0.2, point_count)
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.add_crs(CRS.from_epsg(32631))
        cloud = laspy.LasData(header)
        cloud.x = 500_000 + (point_columns + 0.5) * 10
        cloud.y = 5_200 - (point_rows + 0.5) * 10
        cloud.z = elevations[point_rows, point_columns] + 2 + noise
        cloud.write("lidar.laz")

        # 3/ Execute the Python block directly from the README so code edits are tested automatically
        readme_path = Path(__file__).resolve().parents[1] / "README.md"
        readme = readme_path.read_text(encoding="utf-8")
        quick_start = readme.split("## Quick start\n", 1)[1].split("\n## ", 1)[0]
        code_blocks = re.findall(r"^```python\n(.*?)^```$", quick_start, flags=re.MULTILINE | re.DOTALL)
        assert len(code_blocks) == 1, "Expected one Python block in the README quick start"

        # Capture the README's print() argument, since dependencies can also write to stdout
        readme_print = Mock()
        namespace: dict[str, Any] = {"print": readme_print}

        # Older Xarray versions drop attributes during arithmetic unless we explicitly preserve them
        with xr.set_options(keep_attrs=keep_attrs):
            exec(compile(code_blocks[0], str(readme_path), "exec"), namespace)
        ds, ds_lidar = namespace["ds"], namespace["ds_lidar"]
        ds_point, ds_diff = namespace["ds_point"], namespace["ds_diff"]
        variogram = namespace["variogram"]
        readme_print.assert_called_once()
        statistics = readme_print.call_args.args[0]

        # 4/ Check that sources and outputs stay lazy, including the final three-point chunk
        da = import_optional("dask").array
        assert isinstance(ds.data, da.Array)
        assert not ds._in_memory
        assert ds.chunks == ((512, 8), (512, 8))
        for result in (ds_lidar, ds_point, ds_diff):
            assert isinstance(result.data, da.Array)
            assert not result._in_memory
            assert result.chunks == ((10_000, 3),)
            assert result.sizes["point"] == point_count
            assert result.pc.crs == ds.rst.crs
        assert isinstance(variogram, gu.Variogram)
        assert variogram.model is not None
        assert variogram.model.model_name == "spherical"

        # Compare the computed predictions exactly with the same method and fitted model on eager files
        eager_raster = gu.open_raster("dem.tif")
        eager_points = gu.open_pointcloud("lidar.laz", data_name="Z", load=True)
        expected_points = eager_raster.rst.resample_at_points(
            eager_points,
            resampling="kriging",
            backend="gstools",
            resampling_kwargs={"model": variogram, "neighborhood": gu.operators.GridNeighbours(size=5)},
        )
        computed_points = ds_point.compute()
        computed_difference = ds_diff.compute()
        np.testing.assert_array_equal(computed_points.values, expected_points.values)
        np.testing.assert_array_equal(computed_difference.values, (expected_points - eager_points).values)

        # Kriging at pixel centers reproduces the DEM; LAS rounds elevations to 0.01 m
        # Allow 1e-5 m for interpolation round-off in the float32 source values
        expected_difference = elevations[point_rows, point_columns] - np.asarray(cloud.z)
        np.testing.assert_allclose(computed_difference.values, expected_difference, rtol=0, atol=1e-5)
        expected_statistics = {
            "mean": np.mean(expected_difference),
            "std": np.std(expected_difference),
            "rmse": np.sqrt(np.mean(expected_difference**2)),
        }
        assert statistics.keys() == expected_statistics.keys()
        for name, expected in expected_statistics.items():
            assert statistics[name] == pytest.approx(expected, abs=1e-5)
