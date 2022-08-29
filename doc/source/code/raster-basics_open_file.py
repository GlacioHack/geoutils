"""Example scripts to open and print information about a raster."""
import geoutils as gu

# Fetch an example file
filename = gu.examples.get_path("everest_landsat_b4")

# Open the file
image = gu.Raster(filename)
#### TEXT
information = image.info()
#### TEXT
information = image.info(stats=True)
#### TEXT
with open("file.txt", "w") as fh:
    fh.writelines(information)
