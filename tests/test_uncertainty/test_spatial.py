"""Test observation weighting and propagated uncertainty through public spatial methods."""

import numpy as np
import pandas as pd
import pytest
import rasterio as rio

import geoutils as gu
from geoutils.operators.neighbours import GridNeighbours
from geoutils.operators.reducer import Mean
from geoutils.stats.variography import VariogramModel
from tests.operator_helpers import LocalMeanInterpolator


class TestSpatialPropagation:
    """Test module for weighted estimates, geometric coefficients, and ordinary spatial return types."""

    @pytest.mark.parametrize("engine", ["scipy", "numba"])
    @pytest.mark.parametrize("method", ["nearest", "idw"])
    def test_grid__rectangular_pixel_distances(self, engine: str, method: str) -> None:
        """Checks that adding homogeneous errors preserves each method's distance units on rectangular pixels."""

        # The first point is nearest in coordinate units, while the second is nearest in output-pixel units
        if engine == "numba":
            pytest.importorskip("numba")
        points = gu.PointCloud.from_xyz([0.0, 4.0], [1.0, 0.0], [10.0, 20.0], crs=32631)
        reference = gu.Raster.from_array(np.zeros((1, 1)), rio.transform.from_origin(0, 0, 10, 1), crs=32631)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 0.5)])
        kwargs = {"ref": reference, "resampling": method, "dist_nodata_pixel": 2, "engine": engine}

        expected = points.grid(**kwargs)
        result = points.grid(**kwargs, error_structure=errors)
        summary = gu.uncertainty.propagate(points.grid, error_structure=errors, operation_kwargs=kwargs)

        # Equal observation errors preserve the distance coefficients, and propagation uses the same result
        np.testing.assert_allclose(result.to_nanarray(), expected.to_nanarray(), rtol=1e-14)
        np.testing.assert_array_equal(summary.estimate.to_nanarray(), result.to_nanarray())
        np.testing.assert_allclose(summary.mean.to_nanarray(), result.to_nanarray(), rtol=1e-14)

    def test_resample_at_points__correlated_fit(self) -> None:
        """Checks that a raster window uses correlated errors for both its mean and propagated variance."""

        pytest.importorskip("skgstat")

        # Two-cell neighborhood with correlated errors (equal geometric weights)
        raster = gu.Raster.from_array(np.array([[10.0, 20.0]]), rio.transform.from_origin(0, 1, 1, 1), crs=32631)
        correlation = VariogramModel("spherical", effective_range=5, partial_sill=1)
        errors = gu.ErrorStructure([gu.ErrorComponent("spatial", 2, correlation)])
        operator = Mean(GridNeighbours(offsets=((0, 0), (0, 1))))
        kwargs = {"points": ([0.5], [0.5]), "method": operator, "as_array": True}

        # Compare direct/propagated results with generalized least squares
        result = raster.resample_at_points(**kwargs, error_structure=errors)
        summary = gu.uncertainty.propagate(raster.resample_at_points, error_structure=errors, operation_kwargs=kwargs)
        covariance = errors.to_covariance_matrix(np.array([[0.5, 0.5], [1.5, 0.5]]))
        weights = np.linalg.solve(covariance, np.ones(2))
        weights /= weights.sum()

        assert result == pytest.approx(weights @ [10, 20])
        assert summary.estimate == result
        assert summary.variance == pytest.approx(weights @ covariance @ weights)
        assert summary.variance > 2

    def test_reproject__integer_input_has_fractional_uncertainty(self) -> None:
        """Checks that a fractional error magnitude remains floating point when the source raster stores integers."""

        # Averaging four independent errors of magnitude one gives a standard deviation of one half
        raster = gu.Raster.from_array(
            np.array([[2, 4], [6, 8]], dtype=np.int16), rio.transform.from_origin(0, 2, 1, 1), crs=32631, nodata=-9999
        )
        reference = gu.Raster.from_array(np.zeros((1, 1)), rio.transform.from_origin(0, 2, 2, 2), crs=32631)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])
        kwargs = {"ref": reference, "resampling": Mean()}

        summary = gu.uncertainty.propagate(raster.reproject, error_structure=errors, operation_kwargs=kwargs)

        assert summary.estimate.to_nanarray()[0, 0] == 5
        assert np.issubdtype(summary.std.data.dtype, np.floating)
        assert summary.std.to_nanarray()[0, 0] == 0.5

    def test_reproject__bilinear_keeps_geometric_coefficients(self) -> None:
        """Checks that bilinear interpolation uses geometric weights with spatially correlated errors."""

        pytest.importorskip("skgstat")

        # Target halfway between four cells (bilinear weights 1/4 each)
        raster = gu.Raster.from_array(
            np.array([[0.0, 4.0], [8.0, 12.0]]), rio.transform.from_origin(0, 2, 1, 1), crs=32631
        )
        reference = gu.Raster.from_array(np.zeros((1, 1)), rio.transform.from_origin(0.5, 1.5, 1, 1), crs=32631)
        correlation = VariogramModel("spherical", effective_range=5, partial_sill=1)
        errors = gu.ErrorStructure([gu.ErrorComponent("spatial", 2, correlation)])
        kwargs = {"ref": reference, "resampling": "bilinear"}

        result = raster.reproject(**kwargs, error_structure=errors)
        summary = gu.uncertainty.propagate(raster.reproject, error_structure=errors, operation_kwargs=kwargs)
        centers = np.array([[0.5, 1.5], [1.5, 1.5], [0.5, 0.5], [1.5, 0.5]])
        covariance = errors.to_covariance_matrix(centers)
        weights = np.full(4, 0.25)

        # Check bilinear value and variance from geometric weights
        assert result.to_nanarray()[0, 0] == 6
        np.testing.assert_array_equal(summary.estimate.to_nanarray(), result.to_nanarray())
        assert summary.variance.to_nanarray()[0, 0] == pytest.approx(weights @ covariance @ weights)

    def test_reproject__custom_interpolator_multiband_source_ids(self) -> None:
        """Checks that uncertainty uses each source band's cell IDs once for a custom interpolator."""

        # Give the two bands nine distinct observations with independent measurement errors
        first_band = np.arange(9, dtype=float).reshape(3, 3)
        source = gu.Raster.from_array(
            np.stack((first_band, first_band + 9)),
            rio.transform.from_origin(0, 3, 1, 1),
            crs=32632,
            nodata=-9999,
        )
        reference = gu.Raster.from_array(
            np.zeros((1, 1)),
            rio.transform.from_origin(1, 2, 1, 1),
            crs=32632,
            nodata=-9999,
        )
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", 1)])

        # Both bands use the same three by three window, with distinct IDs 0-8 and 9-17
        nominal = source.reproject(resampling=LocalMeanInterpolator(), ref=reference, error_structure=errors)
        summary = gu.uncertainty.propagate(
            source.reproject,
            error_structure=errors,
            operation_kwargs={"resampling": LocalMeanInterpolator(), "ref": reference},
            n_samples=4,
        )
        np.testing.assert_allclose(nominal.to_nanarray().reshape(-1), [4, 13])
        np.testing.assert_allclose(summary.estimate.to_nanarray().reshape(-1), [4, 13])
        np.testing.assert_array_equal(summary.n_valid.to_nanarray().reshape(-1), [4, 4])


class TestSpatialPropagationChunked:
    """Test module for lazy local uncertainty and stable source IDs across spatial and band chunks."""

    def test_reproject__weighted_band_and_spatial_chunks(self) -> None:
        """Checks that chunked weighted reprojection stays lazy and agrees exactly with eager source IDs."""

        # Different values/errors per band to check source IDs across chunks
        pytest.importorskip("dask")
        from dask.callbacks import Callback

        values = np.arange(60.0, dtype=float).reshape(2, 5, 6)
        raster = gu.Raster.from_array(values, rio.transform.from_origin(0, 5, 1, 1), crs=32631, nodata=-9999)
        reference = gu.Raster.from_array(np.zeros((5, 6)), raster.transform, crs=raster.crs)
        statistics = pd.DataFrame({"std": [1.0, 2.0], "count": [30, 30]}, index=pd.Index([0.0, 1.0], name="band"))
        magnitude = gu.ErrorMagnitude.variable_from_grouped_stats(statistics)
        errors = gu.ErrorStructure([gu.ErrorComponent("measurement", magnitude)])
        predictors = {"band": {source_id: float(source_id // 30) for source_id in range(values.size)}}
        kwargs = {"ref": reference, "resampling": Mean(GridNeighbours(size=3)), "nodata_propagation": "ignore"}
        expected = gu.uncertainty.propagate(
            raster.reproject, error_structure=errors, operation_kwargs=kwargs, predictors=predictors
        )

        # 1 x 2 x 4 chunks (band/Y/X), crossing neighborhoods with shorter edge chunks
        lazy = raster.to_xarray().chunk({"band": 1, "y": 2, "x": 4})
        tasks = []
        with Callback(posttask=lambda *args: tasks.append(args[0])):
            result = gu.uncertainty.propagate(
                lazy.rst.reproject, error_structure=errors, operation_kwargs=kwargs, predictors=predictors
            )
        assert not tasks
        assert hasattr(lazy.data, "compute")
        for quantity in ("estimate", "mean", "std"):
            output = getattr(result, quantity)
            assert hasattr(output.data, "compute")
            np.testing.assert_array_equal(output.compute().values, getattr(expected, quantity).to_nanarray())
