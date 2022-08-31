"""Exemplify uses of crop and reproject."""
import geoutils as gu

large_image = gu.Raster(gu.examples.get_path("everest_landsat_b4"))

smaller_image = gu.Raster(gu.examples.get_path("everest_landsat_b4_cropped"))
# TEXT
large_image_orig = large_image.copy()  # Since it gets modified inplace, we want to keep it to print stats.
large_image.crop(smaller_image)
# TEXT
large_image_reprojected = large_image_orig.reproject(smaller_image, dst_nodata=0)
