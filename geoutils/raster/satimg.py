# Copyright (c) 2025 GeoUtils developers
#
# This file is part of the GeoUtils project:
# https://github.com/glaciohack/geoutils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module provides functionalities for parsing sensor metadata, most often related to satellite imagery.
"""

from __future__ import annotations

import datetime as dt
import os
import re
from collections import abc
from typing import Any, TypedDict

import numpy as np

# Metadata tags used for all
satimg_tags = ["platform", "sensor", "product", "version", "tile_name", "datetime"]


class SatImgDict(TypedDict, total=False):
    """Keys and types of inputs associated with image metadata."""

    # Metadata extract directly from filename
    platform: str
    sensor: str
    product: str
    version: str
    tile_name: str
    datetime: dt.datetime

    # Derivative metadata
    tile_xmin: float
    tile_ymin: float
    tile_xsize: float
    tile_ysize: float


def parse_landsat(gname: str) -> list[Any]:
    """Parse Landsat metadata."""

    lsat_sensor = {"C": "OLI/TIRS", "E": "ETM+", "T": "TM", "M": "MSS", "O": "OLI", "TI": "TIRS"}

    attrs: list[Any] = []
    if len(gname.split("_")[0]) > 15:
        attrs.append(f"Landsat {int(gname[2])}")
        attrs.append(lsat_sensor[gname[1]])
        attrs.append(None)
        attrs.append(None)
        attrs.append(gname[3:6] + gname[6:9])
        year = int(gname[9:13])
        doy = int(gname[13:16])
        attrs.append(dt.datetime.fromordinal(dt.date(year - 1, 12, 31).toordinal() + doy))
    elif re.match("L[COTEM][0-9]{2}", gname.split("_")[0]):
        split_name = gname.split("_")
        attrs.append(f"Landsat {int(split_name[0][2:4])}")
        attrs.append(lsat_sensor[split_name[0][1]])
        attrs.append(None)
        attrs.append(None)
        attrs.append(split_name[2][0:3] + split_name[2][3:6])
        attrs.append(dt.datetime.strptime(split_name[3], "%Y%m%d"))
        attrs.append(attrs[3].date())
    return attrs


def parse_metadata_from_fn(filename: str) -> SatImgDict:
    """
    Parse metadata from filename: platform, sensor, product, version, tile name and datetime.

    :param filename: Filename.
    """

    # Extract basename from full filename
    bname = os.path.splitext(os.path.basename(filename))[0]

    # The attributes correspond in order to: platform, sensor, product, version, tile_name, datetime
    tags = satimg_tags

    # First, we assume that the filename has a form XX_YY.ext
    if "_" in bname:
        spl = bname.split("_")

        # Landsat
        if re.match("L[COTEM][0-9]{2}", spl[0]):
            attrs: tuple[Any, ...] | list[Any] = parse_landsat(bname)
        elif spl[0][0] == "L" and len(spl) == 1:
            attrs = parse_landsat(bname)

        # Sentinel-2
        elif re.match("T[0-9]{2}[A-Z]{3}", spl[0]):
            attrs = ("Sentinel-2", "MSI", None, None, spl[0][1:], dt.datetime.strptime(spl[1], "%Y%m%dT%H%M%S"))

        # ArcticDEM and REMA
        elif spl[0] == "SETSM":
            # For PGC DEMs: if the second part of the name starts with a "s", it is the new nomenclature
            # (starting at "s2s041") with the version first,
            # then the rest
            if spl[1][0] == "s":
                version = spl[1]
                index = 1
            # For backwards-compatibility, this is the old nomenclature
            else:
                version = spl[7]
                index = 0
            attrs = (
                "WorldView",
                spl[index + 1],
                "ArcticDEM/REMA/EarthDEM",
                version,
                None,
                dt.datetime.strptime(spl[index + 2], "%Y%m%d"),
            )

        # SPOT
        elif spl[0] == "SPOT":
            attrs = ("HFS", "SPOT5", None, None, None, dt.datetime.strptime(spl[2], "%Y%m%d"))

        # IceBridge
        elif spl[0] == "IODEM3":
            attrs = ("IceBridge", "DMS", "IODEM3", None, None, dt.datetime.strptime(spl[1] + spl[2], "%Y%m%d%H%M%S"))
        elif spl[0] == "ILAKS1B":
            attrs = ("IceBridge", "UAF-LS", "ILAKS1B", None, None, dt.datetime.strptime(spl[1], "%Y%m%d"))

        # ASTER L1A
        elif spl[0] == "AST" and spl[1] == "L1A":
            attrs = (
                "Terra",
                "ASTER",
                "L1A",
                spl[2][2],
                None,
                dt.datetime.strptime(bname.split("_")[2][3:], "%m%d%Y%H%M%S"),
            )

        # ASTER GDEM
        elif spl[0] == "ASTGTM2":
            attrs = ("Terra", "ASTER", "ASTGTM2", "2", spl[1], None)

        # NASADEM
        elif spl[0] == "NASADEM":
            attrs = ("SRTM", "SRTM", "NASADEM-" + spl[1], "1", spl[2], dt.datetime(year=2000, month=2, day=15))

        # TanDEM-X
        elif spl[0] == "TDM1" and spl[1] == "DEM":
            attrs = ("TanDEM-X", "TanDEM-X", "TDM1", "1", spl[4], None)

        # SRTM v4.1
        elif spl[0] == "srtm":
            attrs = ("SRTM", "SRTM", "SRTMv4.1", None, "_".join(spl[1:]), dt.datetime(year=2000, month=2, day=15))
        else:
            attrs = (None,) * 6

    # Or, if the form is only XX.ext
    # (Only the first versions of SRTM had a naming that... bad (simplified?))
    elif os.path.splitext(os.path.basename(filename))[1] == ".hgt":
        attrs = (
            "SRTM",
            "SRTM",
            "SRTMGL1",
            "3",
            os.path.splitext(os.path.basename(filename))[0],
            dt.datetime(year=2000, month=2, day=15),
        )

    else:
        attrs = (None,) * 6

    dict_meta: SatImgDict = {tags[i]: attrs[i] for i in range(len(tags))}  # type: ignore

    return dict_meta


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

    :param latlon: Latitude and longitude.
    :param latlon_sizes: Sizes of lat/lon tiles corresponding to latitude intervals.
    :param lat_lims: Latitude intervals.

    :returns: Tile name.
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


###################################################
# MAIN FUNCTION THAT WILL BE CALLED BY RASTER CLASS
###################################################


def parse_and_convert_metadata_from_filename(filename: str, silent: bool = False) -> SatImgDict:
    """
    Attempts to pull metadata (e.g., sensor, date information) from fname, and convert to human-usable format.

    Sets platform, sensor, tile, datetime, and date attributes.
    """

    # Get list of metadata
    attrs = parse_metadata_from_fn(filename if filename is not None else "")

    # If no metadata was read, return empty dictionary here
    if all(att is None for att in attrs.values()):
        if not silent:
            print("No metadata could be read from filename.")
        return {}

    # Else, if not silent, print what was read
    for k, v in attrs.items():
        if v is not None:
            if not silent:
                print("Setting " + k + " as " + str(v) + " read from filename.")

    # And convert tile name to human-readable tile extent/size
    supported_tile = ["ASTGTM2", "SRTMGL1", "NASADEM", "TDM1"]
    if attrs["tile_name"] is not None and attrs["product"] is not None and attrs["product"] in supported_tile:
        ymin, xmin, yx_sizes, _ = parse_tile_attr_from_name(attrs["tile_name"], product=attrs["product"])
        tile_attrs = SatImgDict(tile_xmin=xmin, tile_ymin=ymin, tile_xsize=yx_sizes[1], tile_ysize=yx_sizes[0])
        attrs.update(tile_attrs)

    return attrs


def decode_sensor_metadata(input_tags: dict[str, str]) -> SatImgDict:
    """
    Decode sensor metadata from their string values saved on disk in Raster.tags.

    :param input_tags:
    :return:
    """

    converted_sensor_tags = SatImgDict()
    for tag in satimg_tags:
        if tag in input_tags:
            if tag == "datetime":
                as_dt = dt.datetime.strptime(input_tags[tag], "%Y-%m-%d %H:%M:%S")
                converted_sensor_tags.update({tag: as_dt})  # type: ignore
            elif tag in ["tile_xmin", "tile_ymin", "tile_xsize", "tile_ymin"]:
                converted_sensor_tags.update({tag: float(input_tags[tag])})  # type: ignore
            elif isinstance(input_tags[tag], str) and input_tags[tag] == "None":
                converted_sensor_tags.update({tag: None})  # type: ignore
                continue

    return converted_sensor_tags
