"""Example script to load a satellite image."""
import geoutils as gu

filename = gu.examples.get_path("everest_landsat_b4_cropped")

satimg = gu.SatelliteImage(filename)
