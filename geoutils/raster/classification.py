# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES)
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functionalities for classifications of Raster objects"""

import json
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from geoutils.raster import Mask, Raster, RasterType


class ClassificationLayer(ABC):
    def __init__(
        self,
        raster: RasterType | str,
        name: str,
    ):
        """
        Initialize the ClassificationLayer object with common attributes.

        :param raster: The Raster on which classification is applied (geoutils.Raster).
        :param name: Name of the classification layer.
        """
        if isinstance(raster, str):
            self.raster = Raster(raster)
        else:
            self.raster = raster
        self.name = name
        self.class_names: dict[int | list[int], str] | None = None  # Will be set in subclasses
        self.classification_masks: Mask | None = None  # Will store classification result (as geoutils.Mask)
        self.stats_df: pd.DataFrame | None = None  # Will store computed statistics for required classes

    @abstractmethod
    def apply_classification(self) -> None:
        """
        Abstract method to apply the classification logic. Subclasses should implement their specific classification
        logic. This method will create the classification as a :class:`geoutils.Mask` object.
        """
        pass

    def get_stats(
        self, req_stats: str | list[str] | None = None, req_stats_classes: str | list[str] | None = None
    ) -> None:
        """
        Compute the required statistics on the classified pixels.

        :param req_stats: List of required statistics to compute (optional, default is all statistics in
            geoutils.Raster.get_stats).
        :param req_stats_classes: List of classes on which to compute statistics (optional, default is all classes).

        :raise ValueError: if req_stats_classes are not class names.
        """
        if self.classification_masks is None or self.class_names is None:
            raise ValueError("Classification has not been applied yet. Call apply_classification() first.")

        # Convert req_stats and req_stats_classes to list
        if isinstance(req_stats, str):
            req_stats = [req_stats]
        if isinstance(req_stats_classes, str):
            req_stats_classes = [req_stats_classes]

        # Check if req_stats_classes are class names
        if req_stats_classes:
            for req_class in req_stats_classes:
                if req_class not in self.class_names.values():
                    raise ValueError(f"{req_class} is not a class name. Class names are : f{self.class_names.values()}")

        stats_list = []

        # Loop over each class in the classification mask
        for i, (class_idx, class_name) in enumerate(self.class_names.items()):
            if req_stats_classes and class_name not in req_stats_classes:
                continue

            # Compute statistics for the class
            class_stats = self.raster.get_stats(
                stats_name=req_stats, inlier_mask=self.classification_masks[i]  # type: ignore
            )

            # Store class, bin info (if applicable), and stats as a dictionary
            class_info: dict[str, str | int | list[int] | np.floating[Any]] = {
                "class_name": class_name,
                "class_idx": class_idx,
            }
            class_info.update(class_stats)

            stats_list.append(class_info)

            # Convert the list of dictionaries into a pandas DataFrame
        self.stats_df = pd.DataFrame(stats_list)

    def save(self, output_dir: str) -> None:
        """
        Save the classification result, class names, and computed statistics to files.

        :param output_dir: Directory where the output files will be saved.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save classification as a GeoTIFF file
        if self.classification_masks is not None:
            classification_path = os.path.join(output_dir, f"{self.name}.tif")
            self.classification_masks.save(classification_path)
            print("Classification masks saved under", classification_path)

        # Save class names as a JSON file
        if self.class_names is not None:
            class_names_path = os.path.join(output_dir, f"{self.name}_classes.json")
            with open(class_names_path, "w") as f:
                json.dump(self.class_names, f, indent=4)
            print("Class names saved under", class_names_path)

        # Save statistics as CSV file
        if self.stats_df is not None:
            stats_csv_path = os.path.join(output_dir, f"{self.name}_stats.csv")
            self.stats_df.to_csv(stats_csv_path, index=False)
            print("Stats saved under", stats_csv_path)


class RasterClassification(ClassificationLayer):
    def __init__(
        self,
        raster: RasterType | str,
        name: str,
        bins: list[float],
    ) -> None:
        """
        Initialize the RasterClassification object.
        :param raster: The Raster object to classify (only 1-band).
        :param name: The name of the classification.
        :param bins: List of bin edges to classify the raster values (e.g., [0, 10, 20, 50, 100, inf]).
        """
        super().__init__(raster, name)
        # Classification works with 1-band rasters
        if self.raster.count != 1:
            raise ValueError("Raster is multi-band, it has to be a 1-band raster")
        self.bins = bins
        self.class_names = {i: f"[{self.bins[i - 1]}, {self.bins[i]})" for i in range(1, len(self.bins))}

    def apply_classification(self) -> None:
        """
        Apply classification based on the bin ranges. This creates a :class:`geoutils.Mask` object where each
        band represent one class.
        """
        # Use np.digitize to classify the raster values into bins
        class_indices = np.digitize(self.raster.data, bins=self.bins)

        # Create a boolean mask for each class
        masks_array = class_indices != np.arange(1, len(self.bins))[:, None, None]

        # Store classification as a Mask object
        self.classification_masks = Mask.from_array(masks_array, transform=self.raster.transform, crs=self.raster.crs)
