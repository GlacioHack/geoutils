"""
geoutils.satimg provides a toolset for working with satellite data.
"""
from __future__ import annotations

import datetime as dt
import os
import re
import warnings
from collections import abc
from typing import Any

import numpy as np
import rasterio as rio

from geoutils.georaster import Raster, RasterType

lsat_sensor = {"C": "OLI/TIRS", "E": "ETM+", "T": "TM", "M": "MSS", "O": "OLI", "TI": "TIRS"}


def parse_landsat(gname: str) -> list[Any]:
    attrs: list[Any] = []
    if len(gname.split("_")[0]) > 15:
        attrs.append(f"Landsat {int(gname[2])}")
        attrs.append(lsat_sensor[gname[1]])
        attrs.append(None)
        attrs.append(None)
        attrs.append((int(gname[3:6]), int(gname[6:9])))
        year = int(gname[9:13])
        doy = int(gname[13:16])
        attrs.append(dt.datetime.fromordinal(dt.date(year - 1, 12, 31).toordinal() + doy))
    elif re.match("L[COTEM][0-9]{2}", gname.split("_")[0]):
        split_name = gname.split("_")
        attrs.append(f"Landsat {int(split_name[0][2:4])}")
        attrs.append(lsat_sensor[split_name[0][1]])
        attrs.append(None)
        attrs.append(None)
        attrs.append((int(split_name[2][0:3]), int(split_name[2][3:6])))
        attrs.append(dt.datetime.strptime(split_name[3], "%Y%m%d"))
        attrs.append(attrs[3].date())
    return attrs


def parse_metadata_from_fn(fname: str) -> list[Any]:

    bname = os.path.splitext(os.path.basename(fname))[0]

    # assumes that the filename has a form XX_YY.ext
    if "_" in bname:

        spl = bname.split("_")

        # attrs corresponds to: satellite, sensor, product, version, tile_name, datetime
        if re.match("L[COTEM][0-9]{2}", spl[0]):
            attrs: tuple[Any, ...] | list[Any] = parse_landsat(bname)
        elif spl[0][0] == "L" and len(spl) == 1:
            attrs = parse_landsat(bname)
        elif re.match("T[0-9]{2}[A-Z]{3}", spl[0]):
            attrs = ("Sentinel-2", "MSI", None, None, spl[0][1:], dt.datetime.strptime(spl[1], "%Y%m%dT%H%M%S"))
        elif spl[0] == "SETSM":
            attrs = ("WorldView", spl[1], "ArcticDEM/REMA", spl[7], None, dt.datetime.strptime(spl[2], "%Y%m%d"))
        elif spl[0] == "SPOT":
            attrs = ("HFS", "SPOT5", None, None, None, dt.datetime.strptime(spl[2], "%Y%m%d"))
        elif spl[0] == "IODEM3":
            attrs = ("IceBridge", "DMS", "IODEM3", None, None, dt.datetime.strptime(spl[1] + spl[2], "%Y%m%d%H%M%S"))
        elif spl[0] == "ILAKS1B":
            attrs = ("IceBridge", "UAF-LS", "ILAKS1B", None, None, dt.datetime.strptime(spl[1], "%Y%m%d"))
        elif spl[0] == "AST" and spl[1] == "L1A":
            attrs = (
                "Terra",
                "ASTER",
                "L1A",
                spl[2][2],
                None,
                dt.datetime.strptime(bname.split("_")[2][3:], "%m%d%Y%H%M%S"),
            )
        elif spl[0] == "ASTGTM2":
            attrs = ("Terra", "ASTER", "ASTGTM2", "2", spl[1], None)
        elif spl[0] == "NASADEM":
            attrs = ("SRTM", "SRTM", "NASADEM-" + spl[1], "1", spl[2], dt.datetime(year=2000, month=2, day=15))
        elif spl[0] == "TDM1" and spl[1] == "DEM":
            attrs = ("TanDEM-X", "TanDEM-X", "TDM1", "1", spl[4], None)
        elif spl[0] == "srtm":
            attrs = ("SRTM", "SRTM", "SRTMv4.1", None, "_".join(spl[1:]), dt.datetime(year=2000, month=2, day=15))
        else:
            attrs = (None,) * 6

    # if the form is only XX.ext (only the first versions of SRTM had a naming that... bad (simplfied?))
    elif os.path.splitext(os.path.basename(fname))[1] == ".hgt":
        attrs = (
            "SRTM",
            "SRTM",
            "SRTMGL1",
            "3",
            os.path.splitext(os.path.basename(fname))[0],
            dt.datetime(year=2000, month=2, day=15),
        )

    else:
        attrs = (None,) * 6

    return list(attrs)


def parse_tile_attr_from_name(tile_name: str, product: str | None = None) -> tuple[float, float, tuple[int, int], int]:
    """
    Convert tile naming to metadata coordinates based on sensor and product
    by default the SRTMGL1 1x1° tile naming convention to lat, lon (originally SRTMGL1)

    :param tile_name: tile name
    :param product: satellite product

    :returns: lat, lon of southwestern corner
    """
    if product is None or product in ["ASTGTM2", "SRTMGL1", "NASADEM"]:
        ymin, xmin = sw_naming_to_latlon(tile_name)
        yx_sizes = (1, 1)
        epsg = 4326
    elif product in ["TDM1"]:
        ymin, xmin = sw_naming_to_latlon(tile_name)
        # TDX tiling
        if ymin >= 80 or ymin < -80:
            yx_sizes = (1, 4)
        elif ymin >= 60 or ymin < -60:
            yx_sizes = (1, 2)
        else:
            yx_sizes = (1, 1)
        epsg = 4326
    else:
        raise ValueError("Tile naming " + tile_name + " not recognized for product " + str(product))

    return ymin, xmin, yx_sizes, epsg


def sw_naming_to_latlon(tile_name: str) -> tuple[float, float]:

    """
    Get latitude and longitude corresponding to southwestern corner of tile naming (originally SRTMGL1 convention)

    Parsing is robust to lower/upper letters to formats with 2 or 3 digits for latitude (NXXWYYY for
    most existing products, but for example it is NXXXWYYY for ALOS) and to reverted formats (WXXXNYY).

    :param tile_name: name of tile

    :return: latitude and longitude of southwestern corner
    """

    tile_name = tile_name.upper()
    if tile_name[0] in ["S", "N"]:
        if "W" in tile_name:
            lon = -int(tile_name[1:].split("W")[1])
            lat_unsigned = int(tile_name[1:].split("W")[0])
        elif "E" in tile_name:
            lon = int(tile_name[1:].split("E")[1])
            lat_unsigned = int(tile_name[1:].split("E")[0])
        else:
            raise ValueError("No west (W) or east (E) in the tile name")

        if tile_name[0] == "S":
            lat = -lat_unsigned
        else:
            lat = lat_unsigned

    elif tile_name[0] in ["W", "E"]:
        if "S" in tile_name:
            lon_unsigned = int(tile_name[1:].split("S")[0])
            lat = -int(tile_name[1:].split("S")[1])
        elif "N" in tile_name:
            lon_unsigned = int(tile_name[1:].split("N")[0])
            lat = int(tile_name[1:].split("N")[1])
        else:
            raise ValueError("No south (S) or north (N) in the tile name")

        if tile_name[0] == "W":
            lon = -lon_unsigned
        else:
            lon = lon_unsigned

    else:
        raise ValueError("Tile not recognized: should start with south (S), north (N), east (E) or west(W)")

    return lat, lon


def latlon_to_sw_naming(
    latlon: tuple[float, float],
    latlon_sizes: abc.Iterable[tuple[float, float]] = ((1.0, 1.0),),
    lat_lims: abc.Iterable[tuple[float, float]] = ((0.0, 90.1),),
) -> str:
    """
    Convert latitude and longitude to widely used southwestern corner tile naming (originally for SRTMGL1)
    Can account for varying tile sizes, and a dependency with the latitude (e.g., TDX global DEM)

    :param latlon: latitude and longitude
    :param latlon_sizes: sizes of lat/lon tiles corresponding to latitude intervals
    :param lat_lims: latitude intervals

    :returns: tile name
    """

    lon = latlon[1]
    lon = ((lon + 180) % 360) - 180
    lat = latlon[0]
    lat = ((lat + 90) % 180) - 90

    if lat < 0:
        str_lat = "S"
    else:
        str_lat = "N"

    if lon < 0:
        str_lon = "W"
    else:
        str_lon = "E"

    tile_name = None
    lat_lims_list = list(lat_lims)
    latlon_sizes_list = list(latlon_sizes)
    for latlim in lat_lims_list:
        if latlim[0] <= np.abs(lat) < latlim[1]:
            ind = lat_lims_list.index(latlim)
            lat_corner = np.floor(lat / latlon_sizes_list[ind][0]) * latlon_sizes_list[ind][0]
            lon_corner = np.floor(lon / latlon_sizes_list[ind][1]) * latlon_sizes_list[ind][1]
            tile_name = str_lat + str(int(abs(lat_corner))).zfill(2) + str_lon + str(int(abs(lon_corner))).zfill(3)

    if tile_name is None:
        raise ValueError("Latitude intervals provided do not contain the lat/lon coordinates")

    return tile_name


satimg_attrs = ["satellite", "sensor", "product", "version", "tile_name", "datetime"]


class SatelliteImage(Raster):

    date: None | dt.datetime

    def __init__(
        self,
        filename_or_dataset: str | RasterType | rio.io.DatasetReader | rio.io.MemoryFile,
        attrs: list[str] | None = None,
        load_data: bool = True,
        bands: int | list[int] | None = None,
        as_memfile: bool = False,
        read_from_fn: bool = True,
        datetime: dt.datetime | None = None,
        tile_name: str | None = None,
        satellite: str | None = None,
        sensor: str | None = None,
        product: str | None = None,
        version: str | None = None,
        read_from_meta: bool = True,
        fn_meta: str | None = None,
        silent: bool = False,
    ) -> None:

        """
        Load satellite data through the Raster class and parse additional attributes from filename or metadata.

        :param filename_or_dataset: The filename of the dataset.
        :param attrs: Additional attributes from rasterio's DataReader class to add to the Raster object.
           Default list is ['bounds', 'count', 'crs', 'dataset_mask', 'driver', 'dtypes', 'height', 'indexes',
           'name', 'nodata', 'res', 'shape', 'transform', 'width'] - if no attrs are specified, these will be added.
        :param load_data: Load the raster data into the object. Default is True.
        :param bands: The band(s) to load into the object. Default is to load all bands.
        :param as_memfile: open the dataset via a rio.MemoryFile.
        :param read_from_fn: Try to read metadata from the filename
        :param datetime: Provide datetime attribute
        :param tile_name: Provide tile name
        :param satellite: Provide satellite name
        :param sensor: Provide sensor name
        :param product: Provide data product name
        :param version: Provide data version
        :param read_from_meta: Try to read metadata from known associated metadata files
        :param fn_meta: Provide filename of associated metadata
        :param silent: No informative output when trying to read metadata

        :return: A SatelliteImage object (Raster subclass)
        """

        # If SatelliteImage is passed, simply point back to SatelliteImage
        if isinstance(filename_or_dataset, SatelliteImage):
            for key in filename_or_dataset.__dict__:
                setattr(self, key, filename_or_dataset.__dict__[key])
            return
        # Else rely on parent Raster class options (including raised errors)
        else:
            super().__init__(filename_or_dataset, attrs=attrs, load_data=load_data, bands=bands, as_memfile=as_memfile)

        # priority to user input
        self.datetime = datetime
        self.tile_name = tile_name
        self.satellite = satellite
        self.sensor = sensor
        self.product = product
        self.version = version

        # trying to get metadata from separate metadata file
        if read_from_meta and self.filename is not None:
            self.__parse_metadata_from_file(fn_meta)

        # trying to get metadata from filename for the None attributes
        if read_from_fn and self.filename is not None:
            self.__parse_metadata_from_fn(silent=silent)

        self.__get_date()

    def __get_date(self) -> dt.datetime | None:  # type: ignore

        """
        Get date from datetime
        :return:
        """
        if self.datetime is not None:
            self.date = self.datetime.date()  # type: ignore
        else:
            self.date = None

    def __parse_metadata_from_fn(self, silent: bool = False) -> None:

        """
        Attempts to pull metadata (e.g., sensor, date information) from fname, setting sensor, satellite,
        tile, datetime, and date attributes.
        """
        fname = self.filename
        name_attrs = ["satellite", "sensor", "product", "version", "tile_name", "datetime"]
        attrs = parse_metadata_from_fn(fname if fname is not None else "")

        if all(att is None for att in attrs):
            if not silent:
                print("No metadata could be read from filename.")
            return

        for n in name_attrs:
            a = self.__getattribute__(n)
            a_fn = attrs[name_attrs.index(n)]
            if a is None and a_fn is not None:
                if not silent:
                    print("From filename: setting " + n + " as " + str(a_fn))
                setattr(self, n, a_fn)
            elif a is not None and attrs[name_attrs.index(n)] is not None:
                if not silent:
                    print(
                        "Leaving user input of "
                        + str(a)
                        + " for attribute "
                        + n
                        + " despite reading "
                        + str(attrs[name_attrs.index(n)])
                        + "from filename"
                    )

    def __parse_metadata_from_file(self, fn_meta: str | None) -> None:
        warnings.warn(f"Parse metadata from file not implemented. {fn_meta}")

        return None

    def copy(self, new_array: np.ndarray | None = None) -> SatelliteImage:
        new_satimg = super().copy(new_array=new_array)  # type: ignore
        # all objects here are immutable so no need for a copy method (string and datetime)
        # satimg_attrs = ['satellite', 'sensor', 'product', 'version', 'tile_name', 'datetime'] #taken outside of class
        for attrs in satimg_attrs:
            setattr(new_satimg, attrs, getattr(self, attrs))

        return new_satimg
