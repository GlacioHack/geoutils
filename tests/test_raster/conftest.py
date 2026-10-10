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
    """Write a two-band image referenced by GCPs/RPCs and containing a missing data patch."""

    # Uneven dimensions expose shorter final chunks; distinct bands expose band assembly errors
    rows, cols = np.indices((61, 73))
    values = np.stack((rows + cols * 2, rows * 3 - cols)).astype("float32")
    values[:, 20:24, 30:35] = -9999
    # A separate patch in the second band checks GDAL's unified source nodata handling
    values[1, 8:12, 15:19] = -9999
    source = gu.Raster.from_array(np.ma.masked_equal(values, -9999), rio.Affine.identity(), None, nodata=-9999)

    if request.param == "gcp_projected":
        # Create four corner GCPs in UTM near the longitude/latitude of the other cases
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
        # Cross-axis terms rotate and shear the image; four points require a first-order fit
        source.gcps = (points, rio.CRS.from_epsg(32632))
    elif request.param == "gcp_nonlinear":
        # Quadratic GCP coordinates curve the image while remaining smooth and invertible
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
    else:
        # Normalize longitude/latitude to image dimensions and include height sensitivity
        line_denominator = [1.0] + [0.0] * 19
        sample_denominator = [1.0] + [0.0] * 19
        line_numerator = [0.0] * 20
        line_numerator[2], line_numerator[3], line_numerator[7] = -1, 0.02, 0.02
        sample_numerator = [0.0] * 20
        sample_numerator[1], sample_numerator[3], sample_numerator[4] = 1, -0.03, 0.03
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

        if request.param in ("rpc_rational", "gcp_rpc_rational"):
            # Add nonconstant denominators that stay positive across the image footprint
            line_denominator[1], line_denominator[8] = 0.025, 0.01
            sample_denominator[2], sample_denominator[7] = -0.02, 0.015
            # Shift the ground and pixel origins and change scales to test RPC normalization
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

        # Combine normalized coordinates with polynomial or rational line/sample mappings
        source.rpcs = RPC(
            **normalization,
            line_num_coeff=line_numerator,
            line_den_coeff=line_denominator,
            samp_num_coeff=sample_numerator,
            samp_den_coeff=sample_denominator,
        )

    # Add projected, curved GCPs for tests that select between two stored georeferencing methods
    # Their CRS and pixel mapping differ from the RPCs, so using the wrong method changes the result
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

    # File-backed inputs let eager and multiprocessing tests use identical metadata and values
    filename = tmp_path / f"{request.param}.tif"
    source.to_file(filename)
    return gu.Raster(filename)
