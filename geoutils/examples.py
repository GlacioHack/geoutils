"""Utility functions to download and find example data."""
import errno
import os
import shutil
import tarfile
import tempfile
import urllib.request
from distutils.dir_util import copy_tree

# Define the location of the data in the example directory
EXAMPLES_DIRECTORY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples/data"))

# Absolute filepaths to the example files.
FILEPATHS_DATA = {
    "everest_landsat_rgb": os.path.join(EXAMPLES_DIRECTORY, "Everest_Landsat","LE71400412000304SGS00_RGB.tif"),
    "everest_landsat_b4": os.path.join(EXAMPLES_DIRECTORY, "Everest_Landsat","LE71400412000304SGS00_B4.tif"),
    "everest_landsat_b4_cropped": os.path.join(EXAMPLES_DIRECTORY, "Everest_Landsat","LE71400412000304SGS00_B4_cropped.tif"),
    "everest_rgi_outlines": os.path.join(EXAMPLES_DIRECTORY, "Everest_Landsat", "15_rgi60_glacier_outlines.gpkg"),
    "exploradores_aster_dem": os.path.join(EXAMPLES_DIRECTORY, "Exploradores_ASTER", "AST_L1A_00303182012144228_Z.tif"),
    "exploradores_rgi_outlines": os.path.join(EXAMPLES_DIRECTORY, "Exploradores_ASTER", "17_rgi60_glacier_outlines.gpkg")
                 }


def download_examples(overwrite: bool = False):
    """
    Fetch the example files.

    :param overwrite: Do not download the files again if they already exist.
    """
    if not overwrite and all(map(os.path.isfile, list(FILEPATHS_DATA.values()))):
        # print("Datasets exist")
        return

    # Static commit hash to be bumped every time it needs to be.
    commit = "846d35119537cd8d6a1b2ac22ea6b79df881386f"
    # The URL from which to download the repository
    url = f"https://github.com/GlacioHack/geoutils-data/tarball/main#commit={commit}"

    # Create a temporary directory to extract the tarball in.
    temp_dir = tempfile.TemporaryDirectory()
    tar_path = os.path.join(temp_dir.name, "data.tar.gz")

    response = urllib.request.urlopen(url)
    # If the response was right, download the tarball to the temporary directory
    if response.getcode() == 200:
        with open(tar_path, "wb") as outfile:
            outfile.write(response.read())
    else:
        raise ValueError(f"Example data fetch gave non-200 response: {response.status_code}")

    # Extract the tarball
    with tarfile.open(tar_path) as tar:
        tar.extractall(temp_dir.name)

    # Find the first directory in the temp_dir (should only be one) and construct the example data dir paths.
    for dir_name in ['Everest_Landsat', 'Exploradores_ASTER']:
        tmp_dir_name = os.path.join(temp_dir.name, [dirname for dirname in os.listdir(temp_dir.name) if os.path.isdir(os.path.join(temp_dir.name, dirname))][0],
        "data", dir_name)

        # Copy the temporary extracted data to the example directory.
        copy_tree(tmp_dir_name, os.path.join(EXAMPLES_DIRECTORY, dir_name))

def get_path(name: str) -> str:
    """
    Get path of example data
    :param name: Name of test data (listed in xdem/examples.py)
    :return:
    """
    if name in list(FILEPATHS_DATA.keys()):
        download_examples()
        return FILEPATHS_DATA[name]
    else:
        raise ValueError('Data name should be one of "'+'" , "'.join(list(FILEPATHS_DATA.keys()))+'".')

