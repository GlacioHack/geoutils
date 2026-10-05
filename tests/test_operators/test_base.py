"""Test local source data and the evaluation rules shared by operators."""

import numpy as np
import pytest

from geoutils._typing import NDArrayBool, NDArrayNum
from geoutils.operators import LinearCoefficients, LocalData


class TestLocalData:
    """Test module of LocalData class for manipulating local (often neighborhood) arrays."""

    def test_localdata__init(self) -> None:
        """Checks that values, validity and source IDs can be supplied without coordinates or weights."""

        # A simple reduction only needs the observations and their identities
        data = LocalData(values=np.array([2.0, 4.0]), valid=np.array([True, False]), source_ids=np.array([7, 3]))

        # Check the class respects the input observation order (without inventing other metadata)
        np.testing.assert_array_equal(data.values, [2, 4])
        np.testing.assert_array_equal(data.valid, [True, False])
        np.testing.assert_array_equal(data.source_ids, [7, 3])
        assert data.coordinates is None and data.target is None and data.distances is None
        assert data.sample_weights is None and data.support_weights is None and data.interpolation_weights is None

    @pytest.mark.parametrize(
        "field,value",
        [
            ("values", np.ones((2, 1))),
            ("valid", np.ones(1, dtype=bool)),
            ("source_ids", np.arange(3)),
            ("coordinates", np.ones((1, 2))),
            ("target", np.ones((1, 2))),
            ("distances", np.ones(3)),
            ("sample_weights", np.ones(3)),
            ("support_weights", np.ones(3)),
            ("interpolation_weights", np.ones(3)),
        ],
    )
    def test_localdata__init_errors(self, field: str, value: NDArrayNum) -> None:
        """Checks that LocalData() raises errors for arrays with incompatible shapes or numbers of observations."""

        # Replace required or optional array with invalid shape or length
        fields = {"values": np.array([2.0, 4.0]), "valid": np.ones(2, dtype=bool), "source_ids": np.arange(2)}
        fields[field] = value

        # Check error for incompatible arrays
        with pytest.raises(ValueError):
            LocalData(**fields)

    def test_localdata__select_invalid(self) -> None:
        """Checks that select() removes invalid data without affecting IDs or coordinates."""

        # Repeated ID 4 represents repeated use of one observation (IDs 4 and 8 can share coordinates)
        data = LocalData(
            values=np.array([1.0, 2.0, 3.0]),
            valid=np.array([True, False, True]),
            source_ids=np.array([4, 8, 4]),
            coordinates=np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]]),
        )
        selected = data.select(data.valid)

        # Check that the two valid observations have source ID 4 and their original coordinates
        np.testing.assert_array_equal(selected.source_ids, np.array([4, 4]))
        assert selected.coordinates is not None
        np.testing.assert_array_equal(selected.coordinates, np.array([[0.0, 0.0], [1.0, 0.0]]))

    @pytest.mark.parametrize("selection", [np.array([False, True]), np.array([False, False])])
    def test_localdata__select_optional(self, selection: NDArrayBool) -> None:
        """Checks that select() filters all observation arrays and returns the original target coordinates."""

        # Distinct coordinates and weights to identify selected rows
        data = LocalData(
            values=np.array([2.0, 4.0]),
            valid=np.array([True, True]),
            source_ids=np.array([7, 3]),
            coordinates=np.array([[0.0, 0.0], [3.0, 4.0]]),
            target=np.array([0.0, 0.0]),
            distances=np.array([0.0, 5.0]),
            sample_weights=np.array([1.0, 2.0]),
            support_weights=np.array([0.25, 0.75]),
            interpolation_weights=np.array([0.6, 0.4]),
        )

        # Select second observation or empty subset
        selected = data.select(selection)

        # Check selected rows in each array, including empty selections
        for field in (
            "values",
            "valid",
            "source_ids",
            "coordinates",
            "distances",
            "sample_weights",
            "support_weights",
            "interpolation_weights",
        ):
            np.testing.assert_array_equal(getattr(selected, field), getattr(data, field)[selection])

        # Check target coordinates and original input values
        assert selected.target is not None and data.target is not None
        np.testing.assert_array_equal(selected.target, data.target)
        np.testing.assert_array_equal(data.values, [2, 4])

    def test_localdata__select_error_covar(self) -> None:
        """Checks that selecting observations selects the same covariance rows and columns."""

        # Select the first and last observations from a covariance with distinct diagonal entries
        covariance = np.diag([1.0, 4.0, 9.0])
        data = LocalData(np.arange(3.0), np.ones(3, dtype=bool), np.arange(3), error_covariance=covariance)
        selected = data.select(np.array([True, False, True]))

        # Both covariance axes follow the remaining observations
        assert selected.error_covariance is not None
        np.testing.assert_array_equal(selected.error_covariance, np.diag([1.0, 9.0]))

    def test_localdata_select__errors(self) -> None:
        """Checks that select() raises ValueError when the selection is shorter than the number of observations."""

        # Two observations requiring two selection flags
        data = LocalData(values=np.array([2.0, 4.0]), valid=np.ones(2, dtype=bool), source_ids=np.arange(2))

        # Check error for short selection
        with pytest.raises(ValueError, match="one boolean per observation"):
            data.select(np.array([True]))

    @pytest.mark.parametrize(
        "covariance, expected",
        [
            ([[1, 0], [0, 4]], [0.8, 0.2]),
            ([[4, 1], [1, 9]], [8 / 11, 3 / 11]),
            ([[0, 0], [0, 4]], [1, 0]),
            ([[1, 2], [2, 4]], [2, -1]),
        ],
    )
    def test_localdata__precision_weights(self, covariance: list[list[float]], expected: list[float]) -> None:
        """Checks that precision weights account for unequal, correlated, exact, and singular errors."""

        # Perfectly correlated errors of magnitudes one and two cancel with coefficients two and minus one
        data = LocalData(
            np.array([10.0, 20.0]), np.ones(2, dtype=bool), np.arange(2), error_covariance=np.asarray(covariance)
        )
        weights = data.precision_weights(np.ones(2))

        # Every result is unbiased and gives the minimum-variance coefficients for its two observations
        np.testing.assert_allclose(weights, expected, atol=1e-14)
        assert weights.sum() == pytest.approx(1)


class TestLinearCoefficients:
    """Test module for signed weights, default and supplied offsets, and invalid inputs to LinearCoefficients()."""

    def test_linearcoef(self) -> None:
        """Checks that LinearCoefficients() stores signed weights and default or supplied offsets."""

        # Signed weights for a weighted difference
        weights = np.array([-1.0, 2.0])

        # Construct with default and supplied offsets
        default = LinearCoefficients(weights)
        shifted = LinearCoefficients(weights, offset=3)

        # Check stored weights
        np.testing.assert_array_equal(default.weights, weights)
        np.testing.assert_array_equal(shifted.weights, weights)

        # Check default and supplied offsets
        assert default.offset == 0.0
        assert shifted.offset == 3.0

    @pytest.mark.parametrize(
        "weights,offset", [(np.ones((2, 1)), 0), (np.array([np.nan]), 0), (np.array([1.0]), np.inf)]
    )
    def test_linearcoef__errors(self, weights: NDArrayNum, offset: float) -> None:
        """Checks that LinearCoefficients() raises a ValueError for invalid weights or offsets."""

        # Check errors for 2D/NaN weights and infinite offsets
        with pytest.raises(ValueError):
            LinearCoefficients(weights=weights, offset=offset)
