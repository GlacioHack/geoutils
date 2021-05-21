"""Exemplify uses of crop and reproject."""
import geoutils as gu

large_image = gu.Raster(gu.datasets.get_path("landsat_B4"))

smaller_image = gu.Raster(gu.datasets.get_path("landsat_B4_crop"))
# TEXT
large_image_orig = large_image.copy()  # Since it gets modified inplace, we want to keep it to print stats.
large_image.crop(smaller_image)
# TEXT
large_image_reprojected = large_image.reproject(smaller_image)
