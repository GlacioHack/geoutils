import os

import numpy as np
import pandas as pd
import pytest

from geoutils import examples
from geoutils.raster import Mask, Raster, RasterBinning, Segmentation


class TestRasterClassification:
    aster_dem_path = examples.get_path("exploradores_aster_dem")
    landsat_b4_path = examples.get_path("everest_landsat_b4")

    # Initialize one of the raster
    aster_dem = Raster(aster_dem_path)

    # Hand-made raster data (simple 3x3 raster with predefined values)
    mock_data = np.array([[1, 5, 15], [25, 50, 75], [100, 150, 200]])

    # Create a hand-made raster using geoutils.Raster class
    mock_raster = Raster.from_array(mock_data, transform=aster_dem.transform, crs=aster_dem.crs)

    # Define bin edges (for classification)
    mock_bins = [0, 10, 20, 50, 100, np.inf]
    landsat_bins = [0, 50, 100, 150, 200, np.inf]
    aster_bins = [0, 1000, 2000, 3000, np.inf]

    # Initialize the RasterClassification instance
    mock_classifier = RasterBinning(raster=mock_raster, name="mock", bins=mock_bins)
    landsat_classifier = RasterBinning(raster=landsat_b4_path, name="landsat", bins=landsat_bins)
    aster_classifier = RasterBinning(raster=aster_dem, name="aster", bins=aster_bins)

    @pytest.mark.parametrize(
        "classifier_bins_name", [(landsat_classifier, landsat_bins, "landsat"), (aster_classifier, aster_bins, "aster")]
    )  # type: ignore
    def test_initialization(self, classifier_bins_name):
        """
        Test if the RasterClassification initializes correctly.
        """
        classifier, bins, name = classifier_bins_name
        assert isinstance(classifier, RasterBinning)
        assert classifier.name is name
        assert classifier.bins == bins
        assert classifier.class_names == {f"[{bins[i-1]}, {bins[i]})": i for i in range(1, len(bins))}

    def test_apply_mock_classification(self) -> None:
        """
        Test the classification on the mock raster and ensure correct output.
        """
        self.mock_classifier.apply()

        # Check that classification_masks is correctly created
        assert self.mock_classifier.classification_masks is not None
        assert isinstance(self.mock_classifier.classification_masks, Mask)

        # Check the content of the classification masks
        masks_array = self.mock_classifier.classification_masks.data

        # Manually expected mask for each bin
        expected_masks = np.array(
            [
                [[False, False, True], [True, True, True], [True, True, True]],  # Bin [0, 10)
                [[True, True, False], [True, True, True], [True, True, True]],  # Bin [10, 20)
                [[True, True, True], [False, True, True], [True, True, True]],  # Bin [20, 50)
                [[True, True, True], [True, False, False], [True, True, True]],  # Bin [50, 100)
                [[True, True, True], [True, True, True], [False, False, False]],  # Bin [100, inf)
            ],
            dtype=bool,
        )

        assert np.array_equal(masks_array, expected_masks)

    @pytest.mark.parametrize("classifier", [landsat_classifier, aster_classifier])  # type: ignore
    def test_apply_classification(self, classifier: RasterBinning):
        """
        Test the classification on raster and ensure it runs without errors.
        """
        classifier.apply()

        assert classifier.classification_masks is not None
        assert isinstance(classifier.classification_masks, Mask)
        assert classifier.classification_masks.shape == classifier.raster.shape
        assert classifier.classification_masks.transform == classifier.raster.transform
        assert classifier.classification_masks.crs == classifier.raster.crs

    def test_get_stats_mock(self) -> None:
        """
        Test the statistics calculation on the mock raster.
        """
        self.mock_classifier.apply()
        self.mock_classifier.get_stats(req_stats="mean")

        # Check that stats_df is correctly populated
        assert self.mock_classifier.stats_df is not None
        assert isinstance(self.mock_classifier.stats_df, pd.DataFrame)

        # Verify that the statistics are calculated correctly
        stats_df = self.mock_classifier.stats_df
        expected_classes = ["[0, 10)", "[10, 20)", "[20, 50)", "[50, 100)", "[100, inf)"]
        expected_means = [3.0, 15.0, 25.0, 62.5, 150.0]

        for i, class_name in enumerate(expected_classes):
            assert stats_df.iloc[i]["class_name"] == class_name
            assert stats_df.iloc[i]["mean"] == expected_means[i]

    @pytest.mark.parametrize(
        "classifier_req_classes", [(landsat_classifier, None), (aster_classifier, ["[0, 1000)", "[1000, 2000)"])]
    )  # type: ignore
    @pytest.mark.parametrize("req_stats", [None, ["Min", "Max", "Mean"]])  # type: ignore
    def test_get_stats(
        self,
        classifier_req_classes: tuple[RasterBinning, str | list[str] | None],
        req_stats: str | list[str] | None,
    ):
        """
        Test the statistics outputs.
        """
        classifier, req_stats_classes = classifier_req_classes
        classifier.apply()
        classifier.get_stats(req_stats=req_stats, req_stats_classes=req_stats_classes)

        # Check that stats_df is correctly populated
        assert classifier.stats_df is not None
        assert isinstance(classifier.stats_df, pd.DataFrame)

        # Ensure all required statistics are computed for each class
        stats_df = classifier.stats_df
        assert "class_name" in stats_df.columns
        assert all(stat in stats_df.columns for stat in ["Mean", "Max", "Min"])

        # Check that the number of rows in the dataframe matches the number of classes (bins - 1)
        if req_stats_classes is None:
            num_classes = len(classifier.bins) - 1
        else:
            num_classes = len(req_stats_classes)
        assert len(stats_df) == num_classes

        assert stats_df.iloc
        assert isinstance(classifier.class_names, dict)
        for i in range(num_classes):
            assert stats_df.iloc[i]["class_idx"] == classifier.class_names.get(stats_df.iloc[i]["class_name"])

        # Ensure that all rows are populated with valid statistics
        assert not stats_df.isnull().values.any()

    @pytest.mark.parametrize("classifier", [landsat_classifier, aster_classifier])  # type: ignore
    def test_save(self, classifier: RasterBinning):
        """
        Test the save functionality to ensure the classification, class names, and statistics are correctly saved.
        """
        output_dir = "test_raster_classification_output"
        classifier.apply()
        classifier.get_stats()
        classifier.save(output_dir)

        # Check that files were saved
        assert os.path.exists(os.path.join(output_dir, f"{classifier.name}.tif"))
        assert os.path.exists(os.path.join(output_dir, f"{classifier.name}_classes.json"))
        assert os.path.exists(os.path.join(output_dir, f"{classifier.name}_stats.csv"))

        # Clean up
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))
        os.rmdir(output_dir)


class TestSegmentation:
    # Set up a mock raster for testing
    aster_dem_path = examples.get_path("exploradores_aster_dem")
    mock_data = np.array([[1, 5, 15], [25, 50, 75], [100, 150, 200]])

    # Create a hand-made raster using geoutils.Raster class
    aster_dem = Raster(aster_dem_path)
    mock_raster = Raster.from_array(mock_data, transform=aster_dem.transform, crs=aster_dem.crs)

    # Define some mock segmentation masks (as example, a 3x3 mask)
    mock_segmentation_masks = np.array(
        [
            [[True, False, False], [False, True, True], [False, False, False]],  # Mask for class 1
            [[False, True, True], [True, False, False], [True, True, True]],  # Mask for class 2
        ]
    )

    mock_class_names: dict[str, int | list[int]] = {"Class A": 1, "Class B": 2}

    segmentation_classifier = Segmentation(
        raster=mock_raster,
        name="segmentation_test",
        classification_masks=mock_segmentation_masks,
        class_names=mock_class_names,
    )

    def test_initialization(self) -> None:
        """
        Test the SegmentationClassification initialization.
        """

        assert isinstance(self.segmentation_classifier, Segmentation)
        assert self.segmentation_classifier.name == "segmentation_test"
        assert self.segmentation_classifier.raster == self.mock_raster
        assert self.segmentation_classifier.classification_masks is not None
        assert self.segmentation_classifier.classification_masks.shape == (3, 3)  # 2x2 segmentation masks
        assert self.segmentation_classifier.classification_masks.count == 2
        assert self.segmentation_classifier.class_names == self.mock_class_names

    def test_get_stats(self) -> None:
        """
        Test the statistics calculation for segmentation classification.
        """
        self.segmentation_classifier.get_stats(req_stats="mean")

        # Check that stats_df is correctly populated
        assert self.segmentation_classifier.stats_df is not None
        assert isinstance(self.segmentation_classifier.stats_df, pd.DataFrame)

        # Verify the statistics output
        stats_df = self.segmentation_classifier.stats_df
        expected_classes = ["Class A", "Class B"]
        expected_means = [82.5, 42]

        for i, class_name in enumerate(expected_classes):
            assert stats_df.iloc[i]["class_name"] == class_name
            assert stats_df.iloc[i]["mean"] == expected_means[i]
