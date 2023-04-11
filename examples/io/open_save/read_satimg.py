"""
Parsing image metadata
======================

This example demonstrates the instantiation of an image through :class:`~geoutils.SatelliteImage`.
"""

import geoutils as gu

# %%
# We print the filename of our raster that, as often with satellite data, holds metadata information.
filename_geoimg = gu.examples.get_path("everest_landsat_b4")
import os

print(os.path.basename(filename_geoimg))

# %%
# We open it as a geo-image, un-silencing the attribute retrieval to see the parsed data.
img = gu.SatelliteImage(filename_geoimg, silent=False)

# %%
# We have now retrieved the metadata. For the rest, the :class:`~geoutils.SatelliteImage` is a subclass of :class:`~geoutils.Raster`, and behaves similarly.
img
