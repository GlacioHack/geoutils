.. _raster-basics:

Opening a raster file
---------------------

	from GeoUtils import raster_tools
	image = raster_tools.Raster('file.tif')


Basic information about a Raster
--------------------------------

To print information directly to your console:

	print(image)

If you'd like to retrieve a string of information about the raster to be saved
to a variable, output to a text file etc:

	information = image.info()

With added stats:

	information = image.info(stats=True)

Then to write a file:
	
	with open('file.txt', 'w') as fh:
		fh.writelines(information)

Or just print nicely to console:

	print(information)