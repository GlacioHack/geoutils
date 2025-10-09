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

"""Utility functions to download and find example data."""

import os
import shutil
import tarfile
import tempfile
import urllib.request

# Define the location of the data in the example directory
_EXAMPLES_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples/data"))

# Absolute filepaths to the example files.
_FILEPATHS_DATA = {
    "everest_landsat_rgb": os.path.join(_EXAMPLES_DIRECTORY, "Everest_Landsat", "LE71400412000304SGS00_RGB.tif"),
    "everest_landsat_b4": os.path.join(_EXAMPLES_DIRECTORY, "Everest_Landsat", "LE71400412000304SGS00_B4.tif"),
    "everest_landsat_b4_cropped": os.path.join(
        _EXAMPLES_DIRECTORY, "Everest_Landsat", "LE71400412000304SGS00_B4_cropped.tif"
    ),
    "everest_rgi_outlines": os.path.join(_EXAMPLES_DIRECTORY, "Everest_Landsat", "15_rgi60_glacier_outlines.gpkg"),
    "exploradores_aster_dem": os.path.join(
        _EXAMPLES_DIRECTORY, "Exploradores_ASTER", "AST_L1A_00303182012144228_Z.tif"
    ),
    "exploradores_rgi_outlines": os.path.join(
        _EXAMPLES_DIRECTORY, "Exploradores_ASTER", "17_rgi60_glacier_outlines.gpkg"
    ),
    "coromandel_lidar": os.path.join(_EXAMPLES_DIRECTORY, "Coromandel_Lidar", "points.laz"),
}

_FILEPATHS_TEST = {
    k: os.path.join(
        os.path.dirname(v),
        os.path.splitext(os.path.basename(v))[0] + "_test" + os.path.splitext(os.path.basename(v))[1],
    )
    for k, v in _FILEPATHS_DATA.items()
}

available = list(_FILEPATHS_DATA.keys())
available_test = list(_FILEPATHS_TEST.keys())


def download_examples(overwrite: bool = False) -> None:
    """
    Fetch the example files.

    :param overwrite: Do not download the files again if they already exist.
    """
    if not overwrite and all(map(os.path.isfile, list(_FILEPATHS_DATA.values()))):
        # print("Datasets exist")
        return

    # Static commit hash to be bumped every time it needs to be.
    commit = "e758274647a8dd2656d73c3026c90cc77cab8a86"
    # The URL from which to download the repository
    url = f"https://github.com/GlacioHack/geoutils-data/tarball/main#commit={commit}"

    # Create a temporary directory to extract the tarball in.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tar_path = os.path.join(tmp_dir, "data.tar.gz")

        response = urllib.request.urlopen(url)
        # If the response was right, download the tarball to the temporary directory
        if response.getcode() == 200:
            with open(tar_path, "wb") as outfile:
                outfile.write(response.read())
        else:
            raise ValueError(f"Example data fetch gave non-200 response: {response.status_code}")

        # Extract the tarball
        with tarfile.open(tar_path) as tar:
            tar.extractall(tmp_dir)

        # Find the first directory in the temp_dir (should only be one) and construct the example data dir paths.
        for dir_name in ["Everest_Landsat", "Exploradores_ASTER", "Coromandel_Lidar"]:
            tmp_dir_name = os.path.join(
                tmp_dir,
                [dirname for dirname in os.listdir(tmp_dir) if os.path.isdir(os.path.join(tmp_dir, dirname))][0],
                "data",
                dir_name,
            )

            # Copy the temporary extracted data to the example directory.
            shutil.copytree(tmp_dir_name, os.path.join(_EXAMPLES_DIRECTORY, dir_name), dirs_exist_ok=True)


def get_path(name: str) -> str:
    """
    Get path of example data. List of available files can be found in "examples.available".

    :param name: Name of test data.
    :return:
    """
    if name in list(_FILEPATHS_DATA.keys()):
        download_examples()
        return _FILEPATHS_DATA[name]
    else:
        raise ValueError('Data name should be one of "' + '" , "'.join(list(_FILEPATHS_DATA.keys())) + '".')


def get_path_test(name: str) -> str:
    """
    Get path of test data (reduced size). List of available files can be found in "examples.available".

    :param name: Name of test data.
    :return:
    """
    if name in list(_FILEPATHS_TEST.keys()):
        download_examples()
        return _FILEPATHS_TEST[name]
    else:
        raise ValueError('Data name should be one of "' + '" , "'.join(list(_FILEPATHS_TEST.keys())) + '".')
