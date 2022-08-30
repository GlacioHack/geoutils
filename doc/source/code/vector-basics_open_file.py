"""Example script to load a vector file."""
import warnings

warnings.simplefilter("ignore")  # Temporarily filter warnings since something's up with GeoPandas (2021-05-20).
import geoutils as gu

filename = gu.examples.get_path("everest_rgi_outlines")

outlines = gu.Vector(filename)
# TEXT
# Load an example Landsat image
filename = gu.examples.get_path("everest_landsat_b4")
image = gu.Raster(filename)

# Generate a boolean mask from the glacier outlines.
mask = outlines.create_mask(image)
