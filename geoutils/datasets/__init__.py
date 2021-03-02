import os

__all__ = ["datasets", "get_path"]

_module_path = os.path.dirname(__file__)

datasets = {"landsat_B4": "LE71400412000304SGS00_B4_crop.TIF",
            "landsat_B4_crop": "LE71400412000304SGS00_B4_crop2.TIF",
            "landsat_RGB": "LE71400412000304SGS00_RGB.TIF"}


def get_path(dset):
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
    if dset in list(datasets.keys()):
        return os.path.abspath(os.path.join(_module_path, datasets[dset]))
    else:
        msg = "The dataset '{:s}' is not available. ".format(set)
        msg += "Available datasets are {}".format(
            ", ".join(list(datasets.keys())))
        raise ValueError(msg)
