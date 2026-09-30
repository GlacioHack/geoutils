"""Custom operators shared by operator, raster interpolation, gridding and reprojection tests."""

from collections.abc import Sequence

import numpy as np

from geoutils._typing import NDArrayNum
from geoutils.operators import GridNeighbours, Interpolator, LocalData, Reducer
from geoutils.operators.nodata import NodataHandling
from geoutils.operators.reducer import Mean


class WindowRangeInterpolator(Interpolator):
    """Calculate the value range in a three by three raster window."""

    default_neighborhood = GridNeighbours(size=3)

    def __init__(self) -> None:
        """Start with no regular-grid batches evaluated."""

        self.batch_calls = 0

    def predict(self, data: LocalData) -> float:
        """Return the range of valid source values in the source window."""

        return float(np.max(data.values) - np.min(data.values))

    def predict_batch(
        self,
        data: Sequence[LocalData],
        *,
        nodata_propagation: NodataHandling | None = None,
    ) -> NDArrayNum:
        """Count calls to predict_batch() before evaluating each window."""

        self.batch_calls += 1
        return super().predict_batch(data, nodata_propagation=nodata_propagation)


class LocalMeanInterpolator(Interpolator):
    """Average the source values selected for one raster cell or point-cloud target."""

    def predict(self, data: LocalData) -> float:
        """Return the mean of the selected finite source values."""

        return float(np.mean(data.values))


class PropagatingLocalMeanInterpolator(LocalMeanInterpolator):
    """Average nearby values unless the operator's own nodata rule rejects a selected value."""

    default_nodata_propagation = "propagate"


class PropagatingMeanReducer(Mean):
    """Average selected values with a nodata default that public spatial rules can override."""

    default_nodata_propagation = "propagate"


class NoSupportReducer(Reducer):
    """Provide a reducer that deliberately does not accept area weights."""

    def reduce(self, data: LocalData) -> float:
        """Return the sum when no area weights are supplied."""

        return float(np.sum(data.values))


class SupportMassReducer(Reducer):
    """Add the fractions of source cells covered by an output cell."""

    accepts_support_weights = True

    def reduce(self, data: LocalData) -> float:
        """Return the sum of the supplied area fractions."""

        assert data.support_weights is not None
        return float(np.sum(data.support_weights))


class SourceIndexSum(Reducer):
    """Add source cell IDs to reveal changes in identity across raster chunks."""

    def reduce(self, data: LocalData) -> float:
        """Add the IDs of the selected finite cells."""

        return float(np.sum(data.source_ids))
