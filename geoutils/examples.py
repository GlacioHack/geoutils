"""Utility functions to download and find example data."""

import os
import tarfile
import tempfile
import urllib.request
from distutils.dir_util import copy_tree

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
}

available = list(_FILEPATHS_DATA.keys())


def download_examples(overwrite: bool = False) -> None:
    """
    Fetch the example files.

    :param overwrite: Do not download the files again if they already exist.
    """
    if not overwrite and all(map(os.path.isfile, list(_FILEPATHS_DATA.values()))):
        # print("Datasets exist")
        return

    # Static commit hash to be bumped every time it needs to be.
    commit = "3121f37e8de767cb7ea21cbd93b4dd59a81b1ced"
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
        for dir_name in ["Everest_Landsat", "Exploradores_ASTER"]:
            tmp_dir_name = os.path.join(
                tmp_dir,
                [dirname for dirname in os.listdir(tmp_dir) if os.path.isdir(os.path.join(tmp_dir, dirname))][0],
                "data",
                dir_name,
            )

            # Copy the temporary extracted data to the example directory.
            copy_tree(tmp_dir_name, os.path.join(_EXAMPLES_DIRECTORY, dir_name))


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
