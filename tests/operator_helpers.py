"""Custom operators shared by tests of operators, raster resampling at points, point gridding and reprojection."""

from collections.abc import Sequence

import numpy as np

from geoutils._typing import NDArrayNum
from geoutils.operators import GridNeighbours, Interpolator, LocalData, Reducer
from geoutils.operators.nodata import NodataHandling
from geoutils.operators.reducer import Mean


class WindowRangeInterpolator(Interpolator):
    """A simple max-min in a 3x3 window, counting batch calls for tests."""

    default_neighborhood = GridNeighbours(size=3)

    def __init__(self) -> None:
        self.batch_calls = 0

    def predict(self, data: LocalData) -> float:
        return float(np.max(data.values) - np.min(data.values))

    def predict_batch(
        self,
        data: Sequence[LocalData],
        *,
        nodata_propagation: NodataHandling | None = None,
    ) -> NDArrayNum:
        self.batch_calls += 1
        return super().predict_batch(data, nodata_propagation=nodata_propagation)


class LocalMeanInterpolator(Interpolator):
    """A simple local mean for tests."""

    def predict(self, data: LocalData) -> float:
        return float(np.mean(data.values))


class PropagatingLocalMeanInterpolator(LocalMeanInterpolator):
    """Subclass setting the nodata propagation for tests."""

    default_nodata_propagation = "propagate"


class PropagatingMeanReducer(Mean):
    """Subclass setting the nodata propagation for tests."""

    default_nodata_propagation = "propagate"


class NoSupportReducer(Reducer):
    """A reducer that does not support weights for tests."""

    def reduce(self, data: LocalData) -> float:
        return float(np.sum(data.values))


class SupportMassReducer(Reducer):
    """A reducer that supports weights for tests."""

    accepts_support_weights = True

    def reduce(self, data: LocalData) -> float:
        assert data.support_weights is not None
        return float(np.sum(data.support_weights))


class SourceIndexSum(Reducer):
    """A reducer that uses source IDs for tests."""

    def reduce(self, data: LocalData) -> float:
        return float(np.sum(data.source_ids))
