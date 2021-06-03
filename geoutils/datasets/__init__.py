import os

__all__ = ["available", "get_path"]

_module_path = os.path.dirname(__file__)

available = {
    "landsat_B4": "LE71400412000304SGS00_B4_crop.TIF",
    "landsat_B4_crop": "LE71400412000304SGS00_B4_crop2.TIF",
    "landsat_RGB": "LE71400412000304SGS00_RGB.TIF",
    "glacier_outlines": "glacier_outlines.gpkg",
}


def get_path(dset: str) -> str:
    """
    Get the path to the data file.
    Parameters
    ----------
    dset : str
        The name of the dataset. See ``geoutils.datasets.available`` for
        all options.
    Examples
    --------
    >>> geoutils.datasets.get_path("landsat_B4")
    """
    if dset in list(available.keys()):
        return os.path.abspath(os.path.join(_module_path, available[dset]))

    msg = f"The dataset '{dset}' is not available. "
    msg += "Available datasets are {}".format(", ".join(list(available.keys())))
    raise ValueError(msg)
