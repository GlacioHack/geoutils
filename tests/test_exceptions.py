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
