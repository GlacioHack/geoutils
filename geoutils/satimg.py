"""
geoutils.satimg provides a toolset for working with satellite data.
"""
import os
import re
import datetime as dt
import numpy as np
from geoutils.georaster import Raster
import collections

lsat_sensor = {'C': 'OLI/TIRS', 'E': 'ETM+', 'T': 'TM', 'M': 'MSS', 'O': 'OLI', 'TI': 'TIRS'}

def parse_landsat(gname):
    attrs = []
    if len(gname.split('_')) == 1:
        attrs.append(lsat_sensor[gname[1]])
        attrs.append('Landsat {}'.format(int(gname[2])))
        attrs.append((int(gname[3:6]), int(gname[6:9])))
        year = int(gname[9:13])
        doy = int(gname[13:16])
        attrs.append(dt.datetime.fromordinal(dt.date(year - 1, 12, 31).toordinal() + doy))
        attrs.append(attrs[3].date())
    elif re.match('L[COTEM][0-9]{2}', gname.split('_')[0]):
        split_name = gname.split('_')
        attrs.append(lsat_sensor[split_name[0][1]])
        attrs.append('Landsat {}'.format(int(split_name[0][2:4])))
        attrs.append((int(split_name[2][0:3]), int(split_name[2][3:6])))
        attrs.append(dt.datetime.strptime(split_name[3], '%Y%m%d'))
        attrs.append(attrs[3].date())
    return attrs


def parse_tile_attr_from_name(tile_name,product=None):
    """
    Convert tile naming to metadata coordinates based on sensor and product
    by default the SRTMGL1 1x1° tile naming convention to lat, lon (originally SRTMGL1)

    :param tile_name: tile name
    :type tile_name: str
    :param product: satellite product
    :type product: str

    :returns: lat, lon of southwestern corner
    """
    if product in ['ASTGTM2','SRTMGL1','NASADEM']:
        ymin, xmin = sw_naming_to_latlon(tile_name)
        yx_sizes = (1,1)
        epsg = 4326
    elif product in ['TDM1']:
        ymin, xmin = sw_naming_to_latlon(tile_name)
        #TDX tiling
        if ymin >= 80 or ymin < -80:
            yx_sizes = (1,4)
        elif ymin >= 60 or ymin < -60:
            yx_sizes = (1,2)
        else:
            yx_sizes = (1,1)
        epsg = 4326


def sw_naming_to_latlon(tile_name):

    """
    Get latitude and longitude corresponding to southwestern corner of tile naming (originally SRTMGL1 convention)
    parsing is robust to lower/upper letters to formats with 2 or 3 digits for latitude (NXXWYYY for most existing products,
    but for example it is NXXXWYYY for ALOS) and to reverted formats (WXXXNYY).

    :param tile_name: name of tile
    :type tile_name: str

    :return: latitude and longitude of southwestern corner
    :rtype: tuple
    """

    tile_name = tile_name.upper()
    if tile_name[0] in ['S','N']:
        if 'W' in tile_name:
            lon = -int(tile_name[1:].split('W')[1])
            lat_unsigned = int(tile_name[1:].split('W')[0])
        elif 'E' in tile_name:
            lon = int(tile_name[1:].split('E')[1])
            lat_unsigned = int(tile_name[1:].split('E')[0])
        else:
            raise ValueError('No west (W) or east (E) in the tile name')

        if tile_name is 'S':
            lat = -lat_unsigned
        else:
            lat = lat_unsigned
    else:
        raise ValueError('No south (S) or north (N) in the tile name')

    return lat, lon

def latlon_to_sw_naming(latlon,latlon_sizes=((1,1),),lat_lims=((0,90),)):
    """
    Convert latitude and longitude to widely used southwestern corner tile naming (originally for SRTMGL1)
    Can account for varying tile sizes, and a dependency with the latitude (e.g., TDX global DEM)

    :param latlon: latitude and longitude
    :type latlon: collections.abc.Iterable
    :param latlon_sizes: sizes of lat/lon tiles corresponding to latitude intervals
    :type latlon_sizes: collections.abc.Iterable
    :param lat_lims: latitude intervals
    :type lat_lims: collections.abc.Iterable

    :returns: tile name
    :rtype: str
    """

    if latlon[0]<0:
        str_lat = 'S'
    else:
        str_lat = 'N'

    if latlon[1]<0:
        str_lon = 'W'
    else:
        str_lon = 'E'

    tile_name = None
    for latlim in lat_lims:
        if latlim[0] <= latlon[0] < latlim[1]:
            ind = lat_lims.index(latlim)
            lat_corner = np.floor(latlon[0]/latlon_sizes[ind][0])*latlon_sizes[ind][0]
            lon_corner = np.floor(latlon[1]/latlon_sizes[ind][1])*latlon_sizes[ind][1]
            tile_name = str_lat+str(int(abs(lat_corner))).zfill(2)+str_lon+str(int(abs(lon_corner))).zfill(3)

    if tile_name is None:
        raise ValueError('Latitude intervals provided do not contain the lat/lon coordinates')

    return tile_name


class SatelliteImage(Raster):

    def __init__(self, filename, attrs=None, load_data=True, bands=None,
                 as_memfile=False, datetime=None, sensor=None, satellite=None, tile_name=None,fn_meta=None):
        super().__init__(filename, attrs=attrs, load_data=load_data, bands=bands, as_memfile=as_memfile)

        #TODO: maybe the Raster class should have an "original filename" attribute that doesn't get erased during
        # in-memory manipulation for the possibility of parsing metadata a later stage?

        # priority to user input
        self.datetime = datetime
        self.sensor = sensor
        self.satellite = satellite
        self.tile_name = tile_name

        # trying to get metadata from separate metadata file
        if self.filename is not None:
            self.__parse_metadata_from_file(fn_meta)

        # trying to get metadata from filename for the None attributes
        if self.filename is not None:
            self.__parse_metadata_from_fn()

        self.__get_date()

    def __get_date(self):

        """
        Get date from datetime
        :return:
        """
        if self.datetime is not None:
            self.date = self.datetime.date()
        else:
            self.date = None

    def __parse_metadata_from_fn(self):

        """
        Attempts to pull metadata (e.g., sensor, date information) from fname, setting sensor, satellite,
        tile, datetime, and date attributes.
        """

        fname = self.filename
        bname = os.path.splitext(os.path.basename(fname))[0]

        # assumes that the filename has a form XX_YY.ext
        if '_' in bname:

            spl = bname.split('_')

            #let's start with granule product
            if re.match('L[COTEM][0-9]{2}', spl[0]):
                attrs = parse_landsat(bname)
                self.sensor = attrs[0]
                self.satellite = attrs[1]
                self.tile_name = attrs[2]
                self.datetime = attrs[3]
                self.date = attrs[4]
            elif spl[0][0] == 'L' and len(spl) == 1:
                attrs = parse_landsat(bname)
                self.sensor = attrs[0]
                self.satellite = attrs[1]
                self.tile_name = attrs[2]
                self.datetime = attrs[3]
                self.date = attrs[4]
            elif re.match('T[0-9]{2}[A-Z]{3}', spl[0]):
                self.sensor = 'MSI'
                self.satellite = 'Sentinel-2'
                self.tile_name = spl[0][1:]
                self.datetime = dt.datetime.strptime(spl[1], '%Y%m%dT%H%M%S')
                self.date = self.datetime.date()
            elif spl[0] == 'SETSM':
                self.sensor = 'WorldView/GeoEye'
                self.satellite = spl[1]
                self.tile_name = None
                self.datetime = dt.datetime.strptime(spl[2], '%Y%m%d')
            elif spl[0] == 'SPOT':
                self.sensor = 'HFS'
                self.satellite = 'SPOT5'
                self.tile_name = None
                self.datetime = dt.datetime.strptime(spl[2], '%Y%m%d')
            elif spl[0] == 'IODEM3':
                self.sensor = 'DMS'
                self.product = 'IODEM3'
                self.satellite = 'IceBridge'
                self.tile_name = None
                self.datetime = dt.datetime.strptime(spl[1] + spl[2], '%Y%m%d%H%M%S')
            elif spl[0] == 'ILAKS1B':
                self.sensor = 'UAF-LS'
                self.product = 'ILAKS1B'
                self.satellite = 'IceBridge'
                self.tile_name = None
                self.datetime = dt.datetime.strptime(spl[1], '%Y%m%d')
            elif spl[0] == 'AST' and spl[1] == 'L1A':
                self.sensor = 'ASTER'
                self.product = 'L1A'
                self.version = spl[2][2]
                self.satellite = 'Terra'
                self.tile_name = None
                self.datetime = dt.datetime.strptime(bname.split('_')[2][3:], '%m%d%Y%H%M%S')
            elif spl[0] == 'ASTGTM2':
                self.sensor = 'ASTER'
                self.product = 'ASTGTM2'
                self.version = '2'
                self.satellite = 'Terra'
                self.tile_name = spl[1]
                self.datetime = None
            elif spl[0] == 'NASADEM':
                self.sensor = 'SRTM'
                self.version = '1'
                self.product = 'NASADEM-' + spl[1]
                self.satellite = 'SRTM'
                self.tile_name = spl[2]
                self.datetime = dt.datetime(year=2000, month=2, day=15)
            elif spl[0] == 'TDM1' and spl[1] == 'DEM':
                self.sensor = 'TanDEM-X'
                self.version = '1'
                self.product = 'TDM1'
                self.satellite = 'TanDEM-X'
                self.tile_name = spl[-2]
            elif spl[0] == 'srtm':
                self.sensor = 'SRTM'
                self.version = '4.1'
                self.product = 'SRTMv4.1'
                self.satellite = 'SRTM'
                self.tile_name = '_'.join(spl[1:])
            else:
                print("No metadata could be read from filename.")
                self.sensor = None
                self.satellite = None
                self.tile_name = None
                self.datetime = None

        # if the form is only XX.ext
        elif os.path.splitext(os.path.basename(fname))[1] == 'hgt':
            self.sensor = 'SRTM'
            self.version = '3' #don't think the version 2 is still distributed anyway
            self.product = 'SRTMGL1'
            self.satellite = 'SRTM'
            self.tile_name = os.path.splitext(os.path.basename(fname))
            self.datetime = dt.datetime(year=2000, month=2, day=15)

        else:
            print("No metadata could be read from filename.")
            self.sensor = None
            self.satellite = None
            self.tile_name = None
            self.datetime = None


    def __parse_metadata_from_file(self,fn_meta):
        pass


