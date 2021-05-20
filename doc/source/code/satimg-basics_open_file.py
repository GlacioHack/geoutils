"""Example script to load a satellite image."""
import geoutils as gu

filename = gu.datasets.get_path("landsat_B4_crop")

satimg = gu.SatelliteImage(filename)
