"""Tests for estimating error magnitudes and spatial correlations from elevation differences."""

from __future__ import annotations

import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
from affine import Affine
from scipy.ndimage import gaussian_filter

import geoutils as gu
from geoutils.multiproc import MultiprocConfig
from geoutils.stats.variography import VariogramModel


@pytest.fixture
def pointcloud_file(tmp_path: Any) -> Any:
    """Write point fixtures as GeoParquet row groups with their original labels and attribute dtypes."""

    import geopandas as gpd

    pytest.importorskip("pyarrow")
    filenames = []

    def write_points(frame: gpd.GeoDataFrame, partitions: int) -> tuple[str, int, str | None]:
        """Save one point fixture and return its filename, row chunk size and active value column."""

        # Preserve explicit elevation selection, otherwise infer the common point value columns
        frame = frame.copy()
        column = frame.pc.data_name
        if "data_name" not in frame.attrs and column is None:
            column = next((name for name in ("height", "z", "intensity") if name in frame), None)
        if column is None and not frame.geometry.has_z.all():
            frame["z"] = np.zeros(len(frame))
            column = "z"

        # A final shorter row group checks that file readers preserve every row
        chunks = max(1, len(frame) // partitions + 1)
        filename = tmp_path / f"points-{len(filenames)}.parquet"
        filenames.append(filename)
        frame.to_parquet(
            filename, index=True, geometry_encoding="geoarrow", schema_version="1.1.0", row_group_size=chunks
        )
        return str(filename), chunks, column

    return write_points


class TestErrorStructureEstimation:
    """Test module for estimating independent and correlated error components."""

    def test_estimate__constant_independent_component(self) -> None:
        """Checks that independent errors have the standard deviation of the proxy values."""

        # Symmetric point errors have zero median and standard deviation sqrt(2.5)
        values = np.array([-2.0, -1.0, 1.0, 2.0])
        proxy = gu.PointCloud.from_xyz(np.arange(4), np.zeros(4), values, crs=32631)

        # Fit one constant component without a spatial variogram
        structure = gu.ErrorStructure.estimate(
            proxy,
            components={"measurement": {"magnitude": "constant", "correlation": None}},
            spread_estimator=np.std,
            random_state=2,
        )

        # Population variance is the mean of 4, 1, 1, and 4
        assert structure.predict_magnitude() == pytest.approx(np.sqrt(np.mean(values**2)))
        assert structure.empirical_variogram is None

    @pytest.mark.parametrize("kind", ["raster", "point"])
    @pytest.mark.parametrize("other_precision", ["same", "negligible"])
    def test_estimate_error_structure__two_inputs(self, kind: str, other_precision: str) -> None:
        """Checks that two aligned inputs yield the expected error magnitude for either precision assumption."""

        # Measurements differ by known errors; one reference value is missing
        errors = np.array([-2.0, -1.0, 1.0, 2.0])
        reference = np.array([10.0, 10.0, 10.0, np.nan])
        measured = np.full(4, 10.0) + errors
        if kind == "raster":
            transform = Affine(10, 0, 0, 0, -10, 20)
            source = gu.Raster.from_array(measured.reshape(2, 2), transform, 32631, nodata=-9999)
            other = gu.Raster.from_array(reference.reshape(2, 2), transform, 32631, nodata=-9999)
        else:
            x = np.arange(4, dtype=float)
            source = gu.PointCloud.from_xyz(x, np.zeros(4), measured, crs=32631)
            other = gu.PointCloud.from_xyz(x, np.zeros(4), reference, crs=32631)

        # The shared finite support has three errors; equal precision splits their variance in half
        structure = source.estimate_error_structure(
            other,
            other_precision=other_precision,
            components={"measurement": {"magnitude": "constant", "correlation": None}},
            spread_estimator=np.std,
        )

        # Check the magnitude and that no spatial correlation was fitted
        scale = np.sqrt(2) if other_precision == "same" else 1.0
        assert structure.predict_magnitude() == pytest.approx(np.std(errors[:3]) / scale)
        assert structure.empirical_variogram is None

    def test_estimate_independent_variable_component_without_variography(self) -> None:
        """Checks that an independent component recovers increasing magnitudes without fitting a variogram."""

        # Errors with spread increasing with quality
        rng = np.random.default_rng(5)
        predictor = np.broadcast_to(np.linspace(0, 1, 40), (40, 40))
        values = 20 + (0.5 + predictor) * rng.normal(size=predictor.shape)

        # Proxy and predictor on same grid
        transform = Affine(10, 0, 0, 0, -10, 400)
        proxy = gu.Raster.from_array(values, transform, 32632, nodata=-9999)
        predictor_raster = gu.Raster.from_array(predictor, transform, 32632, nodata=-9999)

        # Fit variable magnitude (independent errors)
        structure = gu.ErrorStructure.estimate(
            proxy,
            predictors={"quality": predictor_raster},
            components={"measurement": {"magnitude": "heteroscedastic", "correlation": None}},
            bins=5,
            min_count=50,
            random_state=4,
        )

        # Check increasing error spread with quality
        predicted = structure.predict_magnitude({"quality": np.array([0.1, 0.9])})
        assert predicted[1] > predicted[0]

        # Check no variogram is fitted for independent errors
        assert structure.empirical_variogram is None

    def test_estimate_point_predictor_on_masked_common_support(self) -> None:
        """Checks that point predictor columns and a Boolean mask use the same finite observations."""

        # Point errors with spread set by quality column
        rng = np.random.default_rng(11)
        positions = np.arange(240, dtype=float)
        quality = np.linspace(0, 1, len(positions))
        values = (0.5 + quality) * rng.normal(size=len(positions))
        proxy = gu.PointCloud.from_xyz(positions, np.zeros_like(positions), values, crs=32632)
        proxy.gdf["quality"] = quality

        # Exclude masked points and missing predictor together
        selected = np.ones(len(positions), dtype=bool)
        selected[:40] = False
        proxy.gdf.loc[100, "quality"] = np.nan
        structure = gu.ErrorStructure.estimate(
            proxy,
            predictors={"quality": "quality"},
            components={"measurement": {"magnitude": "heteroscedastic", "correlation": None}},
            mask=selected,
            bins=4,
            min_count=30,
            random_state=5,
        )

        # Check increasing fitted spread on selected rows
        predicted = structure.predict_magnitude({"quality": np.array([0.25, 0.9])})
        assert predicted[1] > predicted[0]

    @pytest.mark.skipif(find_spec("skgstat") is None, reason="Requires scikit-gstat")
    def test_estimate_separates_variable_short_and_fixed_long_components(self) -> None:
        """Checks that estimation separates a variable short range error from a constant long range error."""

        # Random fields with short/long smoothing lengths
        rng = np.random.default_rng(9)
        predictor = np.broadcast_to(np.linspace(0, 1, 48), (48, 48))
        short_field = gaussian_filter(rng.normal(size=predictor.shape), 1)
        long_field = gaussian_filter(rng.normal(size=predictor.shape), 7)

        # Normalize fields before scaling error magnitudes
        short_field = (short_field - short_field.mean()) / short_field.std()
        long_field = (long_field - long_field.mean()) / long_field.std()
        values = (0.5 + 1.5 * predictor) * short_field + 0.7 * long_field

        # Combined errors and quality predictor on same grid
        transform = Affine(10, 0, 0, 0, -10, 480)
        proxy = gu.Raster.from_array(values, transform, 32632, nodata=-9999)
        predictor_raster = gu.Raster.from_array(predictor, transform, 32632, nodata=-9999)

        # Fit default two-component model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            structure = gu.ErrorStructure.estimate(
                proxy,
                predictors={"quality": predictor_raster},
                bins=4,
                min_count=100,
                n_pairs=5_000,
                n_lags=8,
                random_state=2,
            )

        # Check variable short-range and constant long-range models
        short = structure.components["short_range"]
        long = structure.components["long_range"]
        predicted = structure.predict_magnitude({"quality": np.array([0.1, 0.9])})
        assert isinstance(short.magnitude, gu.ErrorMagnitude)
        assert isinstance(long.magnitude, gu.ErrorMagnitude)
        assert isinstance(short.correlation, VariogramModel)
        assert isinstance(long.correlation, VariogramModel)
        assert short.correlation.effective_range is not None and long.correlation.effective_range is not None
        assert short.magnitude.kind == "variable"
        assert long.magnitude.kind == "constant"

        # Check range order and increasing spread
        assert short.correlation.effective_range < long.correlation.effective_range
        assert predicted[1] > predicted[0]

    def test_estimate_error_structure__two_inputs_fit_correlation(self) -> None:
        """Checks that a two-raster fit recovers the same magnitude and correlation as a scaled difference proxy."""

        pytest.importorskip("skgstat")

        # Smooth spatial errors on a common grid, with a constant reference measurement
        y, x = np.mgrid[:12, :12]
        error = np.sin(x / 2) + np.cos(y / 3) + 0.1 * np.random.default_rng(17).normal(size=x.shape)
        transform = Affine(10, 0, 0, 0, -10, 120)
        measured = gu.Raster.from_array(10 + error, transform, 32631, nodata=-9999)
        reference = gu.Raster.from_array(np.full_like(error, 10.0), transform, 32631, nodata=-9999)
        scaled_difference = gu.Raster.from_array(error / np.sqrt(2), transform, 32631, nodata=-9999)
        options: dict[str, Any] = {
            "components": {"spatial": {"magnitude": "constant", "correlation": "spherical"}},
            "spread_estimator": np.std,
            "n_pairs": 300,
            "n_lags": 5,
            "pair_sampling": "random_xy",
            "random_state": 4,
        }

        # Fit both routes under the same equal-precision assumption
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = measured.estimate_error_structure(reference, **options)
            expected = gu.ErrorStructure.estimate(scaled_difference, **options)

        # Check the fitted component and empirical variogram on the same support
        assert result.predict_magnitude() == pytest.approx(expected.predict_magnitude())
        distances = np.array([0.0, 10.0, 40.0])
        np.testing.assert_allclose(result.predict_correlation(distances), expected.predict_correlation(distances))
        assert result.empirical_variogram is not None
        assert expected.empirical_variogram is not None
        np.testing.assert_allclose(result.empirical_variogram.semivariance, expected.empirical_variogram.semivariance)

    def test_estimate_error_structure__eager_raster_accessor_variable_correlation(self) -> None:
        """Checks that an eager DataArray fits the same variable magnitude and correlation as a Raster."""

        pytest.importorskip("skgstat")

        # Two measurements and a quality predictor on the same 12 x 12 grid
        quality = np.broadcast_to(np.linspace(0, 1, 12), (12, 12)).copy()
        errors = (1 + quality) * np.random.default_rng(31).normal(size=quality.shape)
        transform = Affine(10, 0, 0, 0, -10, 120)
        measured = gu.Raster.from_array(10 + errors, transform, 32632, nodata=-9999)
        reference = gu.Raster.from_array(np.full_like(errors, 10.0), transform, 32632, nodata=-9999)
        predictor = gu.Raster.from_array(quality, transform, 32632, nodata=-9999)
        options: dict[str, Any] = {
            "components": {"spatial": {"magnitude": "heteroscedastic", "correlation": "spherical"}},
            "bins": 3,
            "min_count": 10,
            "spread_estimator": np.std,
            "n_pairs": 300,
            "n_lags": 5,
            "pair_sampling": "random_xy",
            "random_state": 4,
        }

        # Fit the same scaled difference through Raster and eager DataArray inputs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            expected = measured.estimate_error_structure(reference, predictors={"quality": predictor}, **options)
            result = measured.to_xarray().rst.estimate_error_structure(
                reference.to_xarray(), predictors={"quality": predictor.to_xarray()}, **options
            )

        # Compare the grouped magnitude table and fitted spatial correlation
        expected_magnitude = expected.components["spatial"].magnitude
        result_magnitude = result.components["spatial"].magnitude
        assert isinstance(expected_magnitude, gu.ErrorMagnitude)
        assert isinstance(result_magnitude, gu.ErrorMagnitude)
        pd.testing.assert_frame_equal(result_magnitude.grouped_statistics, expected_magnitude.grouped_statistics)
        assert result.empirical_variogram is not None
        assert expected.empirical_variogram is not None
        np.testing.assert_allclose(result.empirical_variogram.semivariance, expected.empirical_variogram.semivariance)

    def test_refit__uses_retained_empirical_variogram(self) -> None:
        """Checks that refit() changes the correlation model using retained empirical variogram bins."""

        # Require optional variogram fitting backend
        pytest.importorskip("skgstat")

        # Smooth grid variation at two spatial scales
        y, x = np.mgrid[:40, :40]
        values = np.sin(x / 4) + 0.5 * np.cos(y / 10)
        proxy = gu.Raster.from_array(values, Affine(10, 0, 0, 0, -10, 400), 32632, nodata=-9999)

        # Fit Gaussian correlation, then refit bins with spherical model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            structure = gu.ErrorStructure.estimate(
                proxy,
                components={"spatial": {"magnitude": "constant", "correlation": "gaussian"}},
                n_pairs=1_000,
                n_lags=6,
                random_state=42,
            )
            refitted = structure.refit("spherical")

        # Check the refit uses the retained empirical bins and changes the component model
        assert structure.empirical_variogram is not None
        assert refitted.empirical_variogram is not None
        np.testing.assert_array_equal(refitted.empirical_variogram.lags, structure.empirical_variogram.lags)
        refitted_model = refitted.components["spatial"].correlation
        assert isinstance(refitted_model, VariogramModel)
        assert refitted_model.model_name == "spherical"


class TestErrorStructureEstimationChunked:
    """Test module for Dask/MP estimation across input types and error components."""

    # Vary point representations only for Dask inputs that actually contain points
    @pytest.mark.parametrize(
        ("backend", "input_type", "as_type"),
        [
            ("dask", "raster", "dataarray"),
            ("dask", "point", "dataarray"),
            ("dask", "point", "geodataframe"),
            ("dask", "point-point", "dataarray"),
            ("dask", "point-point", "geodataframe"),
            ("dask", "raster-point", "dataarray"),
            ("dask", "raster-point", "geodataframe"),
            ("dask", "raster-raster", "dataarray"),
            ("multiproc", "raster", "dataarray"),
            ("multiproc", "point", "dataarray"),
            ("multiproc", "point-point", "dataarray"),
            ("multiproc", "raster-point", "dataarray"),
            ("multiproc", "raster-raster", "dataarray"),
        ],
    )
    @pytest.mark.parametrize(
        "components",
        [
            pytest.param({"measurement": {"magnitude": "constant", "correlation": None}}, id="constant"),
            pytest.param({"measurement": {"magnitude": "heteroscedastic", "correlation": None}}, id="variable"),
            pytest.param({"spatial": {"magnitude": "constant", "correlation": "spherical"}}, id="constant-correlation"),
            pytest.param(
                {"spatial": {"magnitude": "heteroscedastic", "correlation": "spherical"}},
                id="variable-correlation",
            ),
            pytest.param(
                {
                    "measurement": {"magnitude": "constant", "correlation": None},
                    "spatial": {"magnitude": "heteroscedastic", "correlation": "spherical"},
                },
                id="constant-plus-variable-correlation",
            ),
        ],
    )
    def test_estimate__chunked_inputs_and_components_match_eager(
        self,
        as_type: Literal["dataarray", "geodataframe"],
        pointcloud_file: Any,
        backend: str,
        input_type: str,
        components: dict[str, dict[str, Any]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Checks that Dask/MP estimates match eager components for raster and point inputs.

        We vary:
         - Dask vs Multiproc (and compare to eager)
         - Input: single raster, single point (when directly an error proxy), or two points, two raster,
           one point and one raster (when another variable to difference with),
         - Error component definition: constant/variable magnitude, with/without correlation, and multiple components.
        """

        from contextlib import ExitStack

        # 1/ Synthetic variables
        rows, columns = np.indices((12, 12))
        quality = columns / 11
        rng = np.random.default_rng(31)
        errors = (1 + quality) * (np.sin(columns / 3) + np.cos(rows / 4) + 0.1 * rng.normal(size=quality.shape))
        correlated = any(item["correlation"] is not None for item in components.values())
        variable = any(item["magnitude"] == "heteroscedastic" for item in components.values())
        # One outlier and NaN
        if not correlated:
            errors[0, 0] = 1000
            errors[0, 1] = np.nan
        reference = 10 + rows / 10 + columns / 20
        source_kind, *other_kinds = input_type.split("-")
        other_kind = other_kinds[0] if other_kinds else None
        measured = reference + errors if other_kind is not None else errors
        transform = Affine(10, 0, 0, 0, -10, 120)
        raster_source = gu.Raster.from_array(measured, transform, 32632, nodata=-9999)
        raster_reference = gu.Raster.from_array(reference, transform, 32632, nodata=-9999)
        raster_predictor = gu.Raster.from_array(quality, transform, 32632, nodata=-9999)
        # Point locations match raster pixel centers for exact gridding
        x, y = raster_source.ij2xy(rows.ravel(), columns.ravel())
        point_source = gu.PointCloud.from_xyz(x, y, measured.ravel(), crs=32632)
        point_source.gdf["quality"] = quality.ravel()
        point_reference = gu.PointCloud.from_xyz(x, y, reference.ravel(), crs=32632)
        source = raster_source if source_kind == "raster" else point_source
        if other_kind == "raster":
            other = raster_reference
        elif other_kind == "point":
            other = point_reference
        else:
            other = None
        eager_predictors = {"quality": raster_predictor if source_kind == "raster" else "quality"} if variable else None
        # We use a smaller sample size than data to also test subsampling
        options: dict[str, Any] = {
            "components": components,
            "bins": 3,
            "min_count": 10,
            "spread_estimator": np.std,
            "subsample_magnitude": 60 if correlated else 1,
            "n_pairs": 300,
            "n_lags": 5,
            "pair_sampling": "random_xy",
            "random_state": 4,
        }

        # We estimate the error structure in memory for comparison later
        if correlated:
            pytest.importorskip("skgstat")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if other is None:
                expected = gu.ErrorStructure.estimate(source, predictors=eager_predictors, **options)
            else:
                expected = source.estimate_error_structure(other, predictors=eager_predictors, **options)

        # 2/ Chunked with Dask/MP, creating the same data lazy/on-file for raster or point
        # We also use an uneven chunk size relative to datasize, to check for edge behaviour
        # Use four raster tiles to limit Dask scheduling overhead
        raster_chunks = {"y": 7, "x": 8}
        chunked_other = None
        chunked_predictor = None
        input_states: list[tuple[Any, str]] = []
        if backend == "dask":
            pytest.importorskip("dask")
            if source_kind == "raster":
                chunked_source = source.to_xarray().chunk(raster_chunks)
                source_interface = chunked_source.rst
            else:
                pytest.importorskip("dask_geopandas")
                from geoutils.pointcloud.pd_accessor import _register_dask_pointcloud_accessor

                _register_dask_pointcloud_accessor()
                # Two partitions of 73 and 71 rows check the shorter final partition
                chunked_source_filename, chunked_source_chunks, chunked_source_column = pointcloud_file(source.gdf, 2)
                chunked_source = gu.open_pointcloud(
                    chunked_source_filename,
                    data_name=chunked_source_column,
                    columns="all",
                    chunks=chunked_source_chunks,
                    as_type=as_type,
                )
                source_interface = chunked_source.pc

            mp_config = None

            if other_kind == "raster":
                chunked_other = other.to_xarray().chunk(raster_chunks)
                input_states.append((chunked_other.rst, "raster"))
            elif other_kind == "point":
                pytest.importorskip("dask_geopandas")
                chunked_other_filename, chunked_other_chunks, chunked_other_column = pointcloud_file(other.gdf, 2)
                chunked_other = gu.open_pointcloud(
                    chunked_other_filename,
                    data_name=chunked_other_column,
                    columns="all",
                    chunks=chunked_other_chunks,
                    as_type=as_type,
                )

                input_states.append((chunked_other.pc, "point"))
            if variable and source_kind == "raster":
                chunked_predictor = raster_predictor.to_xarray().chunk(raster_chunks)
                input_states.append((chunked_predictor.rst, "raster"))
        else:
            # If with MP, from file inputs
            if source_kind == "raster":
                source_path = tmp_path / "source.tif"
                source.to_file(source_path)
                chunked_source = gu.Raster(source_path)
            else:
                chunked_source = source
            source_interface = chunked_source
            mp_config = MultiprocConfig(chunks=(7, 8) if source_kind == "raster" else 23)

            # Raster files must stay unloaded
            if other_kind == "raster":
                other_path = tmp_path / "other.tif"
                other.to_file(other_path)
                chunked_other = gu.Raster(other_path)
                input_states.append((chunked_other, "raster"))
            elif other_kind == "point":
                chunked_other = other
                input_states.append((chunked_other, "point"))
            if variable and source_kind == "raster":
                predictor_path = tmp_path / "quality.tif"
                raster_predictor.to_file(predictor_path)
                chunked_predictor = gu.Raster(predictor_path)
                input_states.append((chunked_predictor, "raster"))

        input_states.insert(0, (source_interface, source_kind))
        for spatial_input, kind in input_states:
            assert spatial_input.is_loaded == (backend == "multiproc" and kind == "point")
        chunked_predictors = (
            {"quality": chunked_predictor if source_kind == "raster" else "quality"} if variable else None
        )

        # For point-point, record MP point difference internally
        submitted_chunks: list[int] = []
        with ExitStack() as stack:
            if backend == "multiproc" and input_type == "point-point" and not variable and not correlated:
                from geoutils.multiproc.cluster import MpCluster
                from geoutils.uncertainty.estimation import _difference_point_partition

                cluster = stack.enter_context(MpCluster({"nb_workers": 2}))
                mp_config = MultiprocConfig(chunks=23, cluster=cluster)
                submit = cluster.submit

                def record_submit(function: Any, *args: Any, **kwargs: Any) -> Any:
                    if function is _difference_point_partition:
                        submitted_chunks.append(len(args[0]))
                    return submit(function, *args, **kwargs)

                monkeypatch.setattr(cluster, "submit", record_submit)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                if chunked_other is None:
                    result = gu.ErrorStructure.estimate(
                        source_interface, predictors=chunked_predictors, mp_config=mp_config, **options
                    )
                else:
                    result = source_interface.estimate_error_structure(
                        chunked_other, predictors=chunked_predictors, mp_config=mp_config, **options
                    )

        # 3/ We check exact equality across Dask/MP
        for spatial_input, kind in input_states:
            assert spatial_input.is_loaded == (backend == "multiproc" and kind == "point")
        if submitted_chunks:
            assert submitted_chunks == [23] * 6 + [5]
        assert result.fit_diagnostics["magnitude"]["valid_count"] == np.count_nonzero(np.isfinite(errors))
        assert list(result.components) == list(expected.components)

        # Compare each component magnitude, including its grouped statistics
        test_quality = {"quality": np.array([0.2, 0.8])} if variable else None
        for name, expected_component in expected.components.items():
            component = result.components[name]
            magnitude = component.magnitude
            eager_magnitude = expected_component.magnitude
            assert isinstance(magnitude, gu.ErrorMagnitude)
            assert isinstance(eager_magnitude, gu.ErrorMagnitude)
            assert magnitude.kind == eager_magnitude.kind
            if magnitude.kind == "variable":
                assert isinstance(magnitude.grouped_statistics, pd.DataFrame)
                assert isinstance(eager_magnitude.grouped_statistics, pd.DataFrame)
                pd.testing.assert_frame_equal(magnitude.grouped_statistics, eager_magnitude.grouped_statistics)
            assert np.allclose(magnitude.predict(test_quality), eager_magnitude.predict(test_quality), equal_nan=True)

            # Compare fitted model parameters for correlated components
            correlation = component.correlation
            eager_correlation = expected_component.correlation
            if eager_correlation is None:
                assert correlation is None
            else:
                assert isinstance(correlation, VariogramModel)
                assert isinstance(eager_correlation, VariogramModel)
                assert correlation.model_name == eager_correlation.model_name
                assert correlation.effective_range == pytest.approx(eager_correlation.effective_range)
                assert correlation.partial_sill == pytest.approx(eager_correlation.partial_sill)

        # Compare the full error structure and empirical variogram
        assert np.allclose(
            result.predict_magnitude(test_quality), expected.predict_magnitude(test_quality), equal_nan=True
        )
        if not correlated:
            assert result.empirical_variogram is None
            assert expected.empirical_variogram is None
        else:
            distances = np.array([0.0, 10.0, 40.0])
            assert np.allclose(
                result.predict_correlation(distances), expected.predict_correlation(distances), equal_nan=True
            )
            assert result.empirical_variogram is not None
            assert expected.empirical_variogram is not None
            np.testing.assert_array_equal(result.empirical_variogram.counts, expected.empirical_variogram.counts)
            assert np.allclose(result.empirical_variogram.lags, expected.empirical_variogram.lags, equal_nan=True)
            assert np.allclose(
                result.empirical_variogram.semivariance, expected.empirical_variogram.semivariance, equal_nan=True
            )


class TestErrorStructureEstimationErrors:
    """Test module for invalid configurations and statistical controls in ErrorStructure.estimate()."""

    @pytest.mark.parametrize("kind", ["raster", "point"])
    def test_estimate_error_structure__error_invalid_precision(self, kind: str) -> None:
        """Checks an error is raised for an unknown second-input precision assumption."""

        # Comparable measurements on the same locations
        values = np.array([-2.0, -1.0, 1.0, 2.0])
        if kind == "raster":
            source = gu.Raster.from_array(values.reshape(2, 2), Affine(1, 0, 0, 0, -1, 2), 32631, nodata=-9999)
        else:
            source = gu.PointCloud.from_xyz(np.arange(4), np.zeros(4), values, crs=32631)

        # Reject assumptions with no defined variance correction
        with pytest.raises(ValueError, match="other_precision must"):
            source.estimate_error_structure(source, other_precision="unknown")

    @pytest.mark.parametrize("proxy_kind", ["raster", "point"])
    @pytest.mark.parametrize(
        "components, error_type, message",
        [
            ({}, ValueError, "at least one named"),
            ({"": {}}, TypeError, "non-empty names"),
            ({"measurement": {"unknown": 1}}, ValueError, "Unknown configuration"),
            ({"measurement": {"magnitude": "invalid"}}, ValueError, "magnitude must"),
            ({"measurement": {"magnitude": "heteroscedastic"}}, ValueError, "requires at least one"),
            ({"measurement": {"correlation": 2}}, TypeError, "variogram model name"),
        ],
    )
    def test_estimate__error_invalid_components(
        self, proxy_kind: str, components: dict[str, dict[str, Any]], error_type: type[Exception], message: str
    ) -> None:
        """Checks an error is raised for invalid components with raster or point inputs."""

        # Synthetic raster/point cloud as error proxy, without anything wrong with it (not the source of error)
        values = np.array([-2.0, -1.0, 1.0, 2.0])
        if proxy_kind == "raster":
            proxy = gu.Raster.from_array(values.reshape(2, 2), Affine(1, 0, 0, 0, -1, 2), 32631, nodata=-9999)
        else:
            proxy = gu.PointCloud.from_xyz(np.arange(4), np.zeros(4), values, crs=32631)

        # Check errors are raised for each invalid case
        with pytest.raises(error_type, match=message):
            gu.ErrorStructure.estimate(proxy, components=components)

    @pytest.mark.parametrize("proxy_kind", ["raster", "point"])
    @pytest.mark.parametrize(
        "options, error_type, message",
        [
            ({"fit_method": "unknown"}, NotImplementedError, "Only fit_method"),
            ({"min_count": 0}, ValueError, "min_count must"),
            ({"spread_estimator": np.min}, ValueError, "spread_estimator returned"),
            (
                {"components": {"first": {"correlation": None}, "second": {"correlation": None}}},
                ValueError,
                "Only one independent component",
            ),
        ],
    )
    def test_estimate__error_invalid_fitting(
        self, proxy_kind: str, options: dict[str, Any], error_type: type[Exception], message: str
    ) -> None:
        """Checks an error is raised for invalid fitting options with raster or point inputs."""

        # Synthetic raster/point cloud as error proxy, without anything wrong with it (not the source of error)
        values = np.array([-2.0, -1.0, 1.0, 2.0])
        if proxy_kind == "raster":
            proxy = gu.Raster.from_array(values.reshape(2, 2), Affine(1, 0, 0, 0, -1, 2), 32631, nodata=-9999)
        else:
            proxy = gu.PointCloud.from_xyz(np.arange(4), np.zeros(4), values, crs=32631)
        default: dict[str, Any] = {"components": {"measurement": {"correlation": None}}, "spread_estimator": np.std}

        # Check error for each case
        with pytest.raises(error_type, match=message):
            gu.ErrorStructure.estimate(proxy, **(default | options))

    def test_estimate__error_raster_predictor_column(self) -> None:
        """Checks an error is raised when a raster predictor is given as a point column name."""

        # Synthetic raster cloud as error proxy, without anything wrong with it (not the source of error)
        values = np.array([[-2.0, -1.0], [1.0, 2.0]])
        proxy = gu.Raster.from_array(values, Affine(1, 0, 0, 0, -1, 2), 32631, nodata=-9999)

        # Check that raster predictors cannot use point column names
        with pytest.raises(TypeError, match="Raster magnitude predictors cannot be column names"):
            gu.ErrorStructure.estimate(
                proxy,
                predictors={"quality": "quality"},
                components={"measurement": {"magnitude": "heteroscedastic", "correlation": None}},
            )
