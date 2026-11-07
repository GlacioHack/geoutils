"""Tests for GeoUtils exceptions: warnings/errors."""

import warnings

import pytest

from geoutils.exceptions import (
    CRSMismatchError,
    GeoUtilsDeprecationWarning,
    GeoUtilsInputError,
    GeoUtilsMismatchError,
    GeoUtilsUserWarning,
    GeoUtilsWarning,
    GridMismatchError,
    IgnoredGridWarning,
    InvalidBoundsError,
    InvalidCRSError,
    InvalidGridError,
    InvalidNodataError,
    InvalidPointsError,
    InvalidResamplingError,
    InvalidResolutionError,
    InvalidShapeError,
    NoValidDataError,
    PointsMismatchError,
    ShapeMismatchError,
    _warn,
)


class TestInputErrors:
    """Test module for catching input errors through package and standard categories."""

    @pytest.mark.parametrize(
        "error",
        [
            GeoUtilsMismatchError,
            CRSMismatchError,
            GridMismatchError,
            PointsMismatchError,
            ShapeMismatchError,
            InvalidBoundsError,
            InvalidCRSError,
            InvalidGridError,
            InvalidNodataError,
            InvalidPointsError,
            InvalidResamplingError,
            InvalidResolutionError,
            InvalidShapeError,
            NoValidDataError,
        ],
    )
    def test_input_errors__inheritance(self, error: type[GeoUtilsInputError]) -> None:
        """Checks that specific input errors can be caught as GeoUtilsInputError or ValueError."""

        assert issubclass(error, GeoUtilsInputError)
        assert issubclass(error, ValueError)

    @pytest.mark.parametrize("error", [CRSMismatchError, GridMismatchError, PointsMismatchError, ShapeMismatchError])
    def test_mismatch_errors__inheritance(self, error: type[GeoUtilsMismatchError]) -> None:
        """Checks that mismatch errors share a category distinct from invalid definitions."""

        assert issubclass(error, GeoUtilsMismatchError)
        assert not issubclass(error, (InvalidCRSError, InvalidGridError, InvalidPointsError, InvalidShapeError))


class TestWarnings:
    """Test module for filtering GeoUtils warnings through package and standard categories."""

    @pytest.mark.parametrize(
        "category",
        [
            GeoUtilsUserWarning,
            GeoUtilsDeprecationWarning,
            UserWarning,
            DeprecationWarning,
            PendingDeprecationWarning,
            ImportWarning,
            FutureWarning,
            RuntimeWarning,
            ResourceWarning,
        ],
    )
    def test_warning_filters__unexpected_warning(self, category: type[Warning]) -> None:
        """Checks that pytest raises an error for an unexpected package or dependency warning."""

        with pytest.raises(category, match="Unexpected warning"):
            warnings.warn("Unexpected warning", category)

    @pytest.mark.parametrize(
        "warning_kind,category,modules,unexpected_module",
        [
            (
                "affine",
                PendingDeprecationWarning,
                (
                    "rasterio.transform",
                    "rioxarray.rioxarray",
                    # Pytest re-emits captured warnings without the package name
                    "transform",
                ),
                None,
            ),
            ("layout", DeprecationWarning, ("laspy.vlrs.known",), "laspy.another_module"),
            (
                "scalar_conversion",
                DeprecationWarning,
                (
                    "pyproj.transformer",
                    # Pytest uses the source filename when it re-emits a captured warning
                    "/site-packages/pyproj/transformer",
                    r"C:\site-packages\pyproj\transformer",
                ),
                "geoutils.projtools",
            ),
            (
                "array_shape",
                DeprecationWarning,
                (
                    "rasterio._io",
                    "rasterio/_io",
                    r"rasterio\_io",
                    "numpy.ma.core",
                    "/site-packages/numpy/ma/core",
                    r"C:\site-packages\numpy\ma\core",
                    "geoutils.raster.raster",
                    "benchmarks.workflows.io",
                ),
                None,
            ),
            *[
                (
                    f"inplace_{method}",
                    DeprecationWarning,
                    (
                        "dask.dataframe.io.parquet.arrow",
                        "/site-packages/dask/dataframe/io/parquet/arrow",
                        r"C:\site-packages\dask\dataframe\io\parquet\arrow",
                    ),
                    "geoutils.pointcloud.parquet",
                )
                for method in ("reset_index", "set_index")
            ],
            (
                "oriented_envelope",
                RuntimeWarning,
                (
                    "shapely.constructive",
                    "/site-packages/shapely/constructive",
                    r"C:\site-packages\shapely\constructive",
                ),
                "geoutils.vector.vector",
            ),
            (
                "oriented_envelope_invalid",
                RuntimeWarning,
                (
                    "shapely.constructive",
                    "/site-packages/shapely/constructive",
                    r"C:\site-packages\shapely\constructive",
                ),
                "geoutils.vector.vector",
            ),
            (
                "shared_paths",
                RuntimeWarning,
                ("shapely.linear", "/site-packages/shapely/linear", r"C:\site-packages\shapely\linear"),
                "geoutils.vector.vector",
            ),
        ],
    )
    def test_warning_filters__known_dependency_warnings(
        self,
        warning_kind: str,
        category: type[Warning],
        modules: tuple[str, ...],
        unexpected_module: str | None,
    ) -> None:
        """Checks that pytest ignores known dependency warnings while other messages and categories remain errors."""

        messages = {
            "affine": "Use `@` matmul instead of `*` mul operator for matrix multiplication",
            "layout": "Due to '_pack_', the 'ExtraBytesStruct' Structure will use memory layout compatible with MSVC.",
            "scalar_conversion": (
                "Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. "
                "Ensure you extract a single element from your array before performing this operation. "
                "(Deprecated NumPy 1.25.)"
            ),
            "oriented_envelope": "divide by zero encountered in oriented_envelope",
            "array_shape": (
                "Setting the shape on a NumPy array has been deprecated in NumPy 2.5.\n"
                "As an alternative, you can create a new view using np.reshape (with copy=False if needed)."
            ),
            **{
                f"inplace_{method}": (
                    f"The inplace keyword in GeoDataFrame.{method} is deprecated and will be removed in a future "
                    "version (PDEP-8).\nSee "
                    "https://pandas.pydata.org/docs/dev/whatsnew/v3.1.0.html#deprecation-inplace "
                    "for more details."
                )
                for method in ("reset_index", "set_index")
            },
            "oriented_envelope_invalid": "invalid value encountered in oriented_envelope",
            "shared_paths": "invalid value encountered in shared_paths",
        }
        message = messages[warning_kind]

        # Check the known warning and unrelated warnings for each module spelling
        for module in modules:
            with warnings.catch_warnings(record=True) as emitted:
                warnings.warn_explicit(message, category, filename="dependency.py", lineno=1, module=module)
            assert not emitted

            for unexpected_category in (UserWarning, FutureWarning):
                with pytest.raises(unexpected_category):
                    warnings.warn_explicit(
                        message, unexpected_category, filename="dependency.py", lineno=1, module=module
                    )

            with pytest.raises(category):
                warnings.warn_explicit(
                    "Unexpected dependency warning", category, filename="dependency.py", lineno=1, module=module
                )

        # Check module restrictions where the filter defines one
        if unexpected_module is not None:
            with pytest.raises(category):
                warnings.warn_explicit(message, category, filename="dependency.py", lineno=1, module=unexpected_module)

    @pytest.mark.parametrize(
        "category, standard_category",
        [
            (GeoUtilsUserWarning, UserWarning),
            (GeoUtilsDeprecationWarning, DeprecationWarning),
            (IgnoredGridWarning, UserWarning),
        ],
    )
    def test_warning_categories__filters(
        self, category: type[GeoUtilsWarning], standard_category: type[Warning]
    ) -> None:
        """Checks that GeoUtils warnings match their standard category and a shared package filter."""

        with pytest.warns(standard_category) as emitted:
            _warn("GeoUtils warning", category)
        assert emitted[0].category is category
        assert issubclass(category, UserWarning) == (standard_category is UserWarning)

        with warnings.catch_warnings():
            warnings.simplefilter("error", GeoUtilsWarning)
            with pytest.raises(category, match="GeoUtils warning"):
                _warn("GeoUtils warning", category)
