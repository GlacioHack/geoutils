#!/usr/bin/env python
"""
Geoviewer provides a command line tool for plotting raster and vector data.

TO DO:
- include some options from imviewer: https://github.com/dshean/imview/blob/master/imview/imviewer.py
"""
from __future__ import annotations

import argparse
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from geoutils.raster import Raster


def getparser() -> argparse.ArgumentParser:
    # Set up description
    parser = argparse.ArgumentParser(
        description="Visualisation tool for any image supported by GDAL. For single band plots (Single band rasters or with option -band) the image will be rendered as a pseudocolor image using the set or default colormap. For 3 or 4 band data, the image will be plotted as an RGB(A) image. For other band counts, an error will be raised and the option -band must be used."
    )

    # Positional arguments
    parser.add_argument("filename", type=str, help="str, path to the image")

    # optional arguments
    parser.add_argument(
        "-cmap",
        dest="cmap",
        type=str,
        default="default",
        help="str, a matplotlib colormap string (default is from rcParams). This parameter is ignored for multi-band rasters.",
    )
    parser.add_argument(
        "-vmin",
        dest="vmin",
        type=str,
        default=None,
        help=(
            "float, the minimum value for colorscale, or can be expressed as a "
            "percentile e.g. 5%% (default is calculated min value). This parameter is ignored for multi-band rasters."
        ),
    )
    parser.add_argument(
        "-vmax",
        dest="vmax",
        type=str,
        default=None,
        help=(
            "float, the maximum value for colorscale, or can be expressed as a "
            "percentile e.g. 95%% (default is calculated max value). This parameter is ignored for multi-band rasters."
        ),
    )
    parser.add_argument(
        "-band",
        dest="band",
        type=int,
        default=None,
        help="int, for multiband images, which band to display. Starts at 1. (Default is to load all bands and display as rasterio, i.e. asuming RGB(A) and with clipping outside [0-255] for int, [0-1] for float).",
    )
    parser.add_argument(
        "-nocb",
        dest="nocb",
        help="If set, will not display a colorbar (Default is to display the colorbar for single-band raster). This parameter is ignored for multi-band rasters.",
        action="store_false",
    )
    parser.add_argument(
        "-clabel", dest="clabel", type=str, default="", help="str, the label for the colorscale (Default is empty)."
    )
    parser.add_argument("-title", dest="title", type=str, default="", help="str, figure title (Default is empty).")
    parser.add_argument(
        "-figsize",
        dest="figsize",
        type=str,
        default="default",
        help=(
            "str, figure size, must be a tuple of size 2, either written with quotes, "
            "or two numbers separated by comma, no space (Default is from rcParams)."
        ),
    )
    parser.add_argument(
        "-max_size",
        dest="max_size",
        type=int,
        default=2000,
        help="int, image size is limited to max_size**2 for faster reading/displaying (Default is 2000).",
    )
    parser.add_argument(
        "-save",
        dest="save",
        type=str,
        default="",
        help="str, filename to the output filename to save to disk (Default is displayed on screen).",
    )
    parser.add_argument(
        "-dpi",
        dest="dpi",
        type=str,
        default="default",
        help="int, dpi value to use when saving figure (Default is from rcParams).",
    )
    parser.add_argument(
        "-nodata",
        dest="nodata",
        type=str,
        default="default",
        help="float, no data value (Default is read from file metadata).",
    )
    parser.add_argument(
        "-noresampl",
        dest="noresampl",
        default=False,
        action="store_true",
        help="True or False, if False then allow dynamic image downscaling, if True, prevent it.",
    )

    return parser


def main(test_args: Sequence[str] = None) -> None:
    # Parse arguments
    parser = getparser()
    args = parser.parse_args(test_args)  # type: ignore

    # Load image metadata
    img = Raster(args.filename, load_data=False)

    # Resample if image is too large
    if ((img.width > args.max_size) or (img.height > args.max_size)) & (not args.noresampl):
        dfact = max(int(img.width / args.max_size), int(img.height / args.max_size))
        print(f"Image will be downsampled by a factor {dfact}.")
    else:
        dfact = 1

    # Read image
    img = Raster(args.filename, downsample=dfact, bands=args.band)

    # Set no data value
    if args.nodata == "default":
        pass
    else:
        try:
            nodata = float(args.nodata)
        except ValueError:
            raise ValueError("Nodata must be a float, currently set to %s" % args.nodata)

        img.set_nodata(nodata)

    # Set default parameters

    # vmin
    if args.vmin is not None:
        try:
            vmin: float | None = float(args.vmin)
        except ValueError:  # Case is not a number
            perc, _ = args.vmin.split("%")
            try:
                perc = float(perc)
                vmin = np.percentile(img.data.compressed(), perc)
            except ValueError:  # Case no % sign
                raise ValueError("vmin must be a float or percentage, currently set to %s" % args.vmin)

    else:
        vmin = None

    # vmax
    if args.vmax is not None:
        try:
            vmax: float | None = float(args.vmax)
        except ValueError:  # Case is not a number
            perc, _ = args.vmax.split("%")
            try:
                perc = float(perc)
                vmax = np.percentile(img.data.compressed(), perc)
            except ValueError:  # Case no % sign
                raise ValueError("vmax must be a float or percentage, currently set to %s" % args.vmax)

    else:
        vmax = None

    # color map
    # Get the list of existing color maps, including reversed maps
    mpl_cmap_list = list(plt.cm.datad.keys())
    mpl_cmap_list.extend([cmap + "_r" for cmap in mpl_cmap_list])

    if args.cmap == "default":
        cmap = plt.rcParams["image.cmap"]
    elif args.cmap in mpl_cmap_list:
        cmap = args.cmap
    else:
        raise ValueError("Wrong cmap, must be in: {}".format(",".join(str(elem) for elem in mpl_cmap_list)))

    # Figsize
    if args.figsize == "default":
        figsize = plt.rcParams["figure.figsize"]
    else:
        try:
            figsize = tuple(int(arg) for arg in args.figsize.split(","))
        except Exception:
            raise ValueError("Figsize must be a tuple of size 2, currently set to %s" % args.figsize)

    # dpi
    if args.dpi == "default":
        dpi = plt.rcParams["figure.dpi"]
    else:
        try:
            dpi = int(args.dpi)
        except ValueError:
            raise ValueError("dpi must be an integer, currently set to %s" % args.dpi)

    # Plot data

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # plot
    img.plot(
        ax=ax,
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        add_cbar=args.nocb,
        cbar_title=args.clabel,
        title=args.title,
    )

    plt.tight_layout()

    # Save
    if args.save != "":
        plt.savefig(args.save, dpi=dpi)
        print("Figure saved to file %s." % args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
