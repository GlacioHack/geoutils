---
file_format: mystnb
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: geoutils-env
  language: python
  name: geoutils
---
(classification)=

# Classification

The classification module provides tools for classifying {class}`~geoutils.Raster` object into discrete classes.
This can be used to categorize continuous data (e.g., elevation, slope) or to group raster pixels based on predefined
categories (e.g., land-use, vegetation types).

Classification is achieved through the {class}`~geoutils.raster.ClassificationLayer` abstract class, which provides the
foundation for implementing specific classification workflows.

## The {class}`~geoutils.raster.ClassificationLayer` class
{class}`~geoutils.raster.ClassificationLayer` is the base class for all classification layers.
It provides methods for setting up raster classification, computing statistics for each class, and saving classification results.

### Methods:
- {func}`~geoutils.raster.ClassificationLayer.get_stats`: Calculates statistics using {func}`~geoutils.Raster.get_stats` for the classified raster data.
- {func}`~geoutils.raster.ClassificationLayer.save`: Saves the classification result, class names, and statistics to the specified directory:
  - The classification mask is saved as a `.tif` file.
  - Class names are saved as a `.json` file.
  - Computed statistics are saved as a `.csv` file.

## The {class}`~geoutils.raster.RasterClassification` class
{class}`~geoutils.raster.RasterClassification` is a specific implementation of {class}`~geoutils.raster.ClassificationLayer`
designed for classifying continuous raster data using binning. This class allows you to define ranges (bins) for continuous raster values
and classify the pixels into discrete classes based on these ranges.

### Example Workflow:
**1. Load Raster:** Load a raster file (e.g., a Digital Elevation Model).

**2. Define Bins:** Set the bin ranges to classify the raster data (e.g., elevation ranges).

**3. Apply Classification:** Classify the raster data based on the bin ranges.

**4. Compute Statistics:** Compute various statistics for each class.

**5. Save Results:** Save the classification and statistics to files.

```{code-cell} ipython3
import numpy as np
import geoutils as gu
from geoutils.raster import RasterClassification

# load raster (note, the raster parameter can also be a file path)
raster_file = gu.examples.get_path("exploradores_aster_dem")
raster = gu.Raster(raster_file)

# Define bins
bins = [0, 1000, 2000, 3000, np.inf]

# Create RasterClassification object
classification = RasterClassification(raster, "elevation", bins)

# Apply the classification
classification.apply_classification()

# Compute statistics
classification.get_stats(req_stats_classes=["[0, 1000)", "[3000, inf)"])

# Save results
classification.save("elevation_classification")
```

```{code-cell} ipython3
:tags: [remove-cell]
import os
os.remove("elevation_classification/elevation.tif")
os.remove("elevation_classification/elevation_classes.json")
os.remove("elevation_classification/elevation_stats.csv")
os.rmdir("elevation_classification")
```

```{code-cell} ipython3
# print stats
print(classification.stats_df)
```

### Methods:
- {func}`~geoutils.raster.RasterClassification.apply_classification` classifies the raster based on the specified bins.
It assigns each pixel to the appropriate class based on which bin the pixel value falls into.
The resulting classification is stored as a multi-band {class}`~geoutils.Mask` object.
