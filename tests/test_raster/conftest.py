"""Shared raster fixtures for GCP/RPC georeferencing tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio as rio
from rasterio.control import GroundControlPoint
from rasterio.rpc import RPC

import geoutils as gu


@pytest.fixture(params=["gcp_projected", "gcp_nonlinear", "rpc_polynomial", "rpc_rational"])
def raster_gcp_rpc(request: Any, tmp_path: Path) -> gu.Raster:
    """For tests, we write a synthetic 2-band image referenced by GCPs/RPCs and with some NaNs."""

    # We use uneven dimensions (to potentially expose issues with shorter final chunks) and
    # two bands with different NaNs (to check nodata behaviour consistency)
    rows, cols = np.indices((61, 73))
    values = np.stack((rows + cols * 2, rows * 3 - cols)).astype("float32")
    values[:, 20:24, 30:35] = -9999
    values[1, 8:12, 15:19] = -9999
    source = gu.Raster.from_array(np.ma.masked_equal(values, -9999), rio.Affine.identity(), None, nodata=-9999)

    # 1/ First, a 4-corner GCPs in UTM CRS to get a linear GCP
    if request.param == "gcp_projected":
        origin_x, origin_y = rio.warp.transform(4326, 32632, [10], [50])
        points = [
            GroundControlPoint(
                row=row,
                col=col,
                x=origin_x[0] + col * 70 + row * 18,
                y=origin_y[0] + col * 12 - row * 110,
                z=0,
            )
            for row in (0, source.height)
            for col in (0, source.width)
        ]
        source.gcps = (points, rio.CRS.from_epsg(32632))

    # 2/ Second, a non-linear quadratic GCP coords which curve the image (while remaining smooth and invertible)
    elif request.param == "gcp_nonlinear":
        points = [
            GroundControlPoint(
                row=row,
                col=col,
                x=10 + col * 0.001 + row * col * 0.000001,
                y=50 - row * 0.001 + col * col * 0.0000004,
                z=0,
            )
            for row in (0, 30, 61)
            for col in (0, 36, 73)
        ]
        source.gcps = (points, rio.CRS.from_epsg(4326))

    # Otherwise, we build a synthetic RPC
    else:
        # 3/ Third, a polynomial RPC (each denominator is 1, so image rows/cols come from polynomials)
        # Normalize lon/lat to image dimensions, and include height sensitivity
        line_denominator = [1.0] + [0.0] * 19
        sample_denominator = [1.0] + [0.0] * 19

        # Height terms make RPC_HEIGHT/RPC_DEM affect pixel positions, quadratic terms curve the image
        line_numerator = [0.0] * 20
        line_numerator[2], line_numerator[3], line_numerator[7] = -1, 0.02, 0.02
        sample_numerator = [0.0] * 20
        sample_numerator[1], sample_numerator[3], sample_numerator[4] = 1, -0.03, 0.03

        # We define ground/image centers and scales to work with coordinates near -1 to 1
        normalization = {
            "height_off": 0.0,
            "height_scale": 100.0,
            "lat_off": 50.0,
            "lat_scale": 0.03,
            "long_off": 10.0,
            "long_scale": 0.036,
            "line_off": 30.0,
            "line_scale": 30.0,
            "samp_off": 36.0,
            "samp_scale": 36.0,
        }

        # 4/ Fourth, a rational RPC (ratios of polynomials describe image rows/cols)
        if request.param in ("rpc_rational", "gcp_rpc_rational"):
            # We make the denominators depend on lon/lat while staying positive (to avoid division by zero)
            line_denominator[1], line_denominator[8] = 0.025, 0.01
            sample_denominator[2], sample_denominator[7] = -0.02, 0.015

            # We also shift the ground/image centers and change the scales to check different RPC normalizations
            normalization.update(
                height_off=30.0,
                height_scale=150.0,
                lat_off=49.97,
                lat_scale=0.025,
                long_off=10.04,
                long_scale=0.032,
                line_off=33.0,
                line_scale=26.0,
                samp_off=31.0,
                samp_scale=40.0,
            )

        # We combine the coefficients and offsets/scales into RPC equations for image rows/cols
        source.rpcs = RPC(
            **normalization,
            line_num_coeff=line_numerator,
            line_den_coeff=line_denominator,
            samp_num_coeff=sample_numerator,
            samp_den_coeff=sample_denominator,
        )

    # 5/ Fifth, GCPs and RPCs together, to check method selection in reproject() when both exist
    if request.param in ("gcp_rpc_polynomial", "gcp_rpc_rational"):
        origin_x, origin_y = rio.warp.transform(4326, 32632, [10], [50])
        points = [
            GroundControlPoint(
                row=row,
                col=col,
                x=origin_x[0] + col * 70 + row * 18 + row * col * 0.1,
                y=origin_y[0] + col * 12 - row * 110 + col * col * 0.04,
                z=0,
            )
            for row in (0, 30, source.height)
            for col in (0, 36, source.width)
        ]
        source.gcps = (points, rio.CRS.from_epsg(32632))

    # Write to file (to test Dask/MP in addition to in-memory)
    filename = tmp_path / f"{request.param}.tif"
    source.to_file(filename)
    return gu.Raster(filename)
