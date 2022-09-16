"""Basic operations to be run on 2D image arrays

Optional dependencies:
    skimage.transform (@subdivide_array)

"""
from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np
import rasterio as rio
import rasterio.warp
from tqdm import tqdm

import geoutils as gu
from geoutils.georaster import Raster, RasterType
from geoutils.georaster.raster import _default_nodata
from geoutils.misc import resampling_method_from_str


def get_mask(array: np.ndarray | np.ma.masked_array) -> np.ndarray:
    """
    Return the mask of invalid values, whether array is a ndarray with NaNs or a np.ma.masked_array.

    :param array: Input array.

    :returns invalid_mask: boolean array, True where array is masked or Nan.
    """
    mask = (array.mask | ~np.isfinite(array.data)) if isinstance(array, np.ma.masked_array) else ~np.isfinite(array)
    return mask.squeeze()


def get_array_and_mask(
    array: np.ndarray | np.ma.masked_array | RasterType, check_shape: bool = True, copy: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return array with masked values set to NaN and the associated mask.
    Works whether array is a ndarray with NaNs or a np.ma.masked_array.

    :param array: Input array.
    :param check_shape: Validate that the array is either a 1D array, a 2D array or a 3D array of shape (1, rows, cols).
    :param copy: Return a copy of 'array'. If False, a view will be attempted (and warn if not possible)

    :returns array_data, invalid_mask: a tuple of ndarrays. First is array with invalid pixels converted to NaN, \
    second is mask of invalid pixels (True if invalid).
    """
    if isinstance(array, gu.Raster):
        array = array.data

    if check_shape:
        if len(array.shape) > 2 and array.shape[0] > 1:
            raise ValueError(
                f"Invalid array shape given: {array.shape}." "Expected 2D array or 3D array where arr.shape[0] == 1"
            )

    # If an occupied mask exists and a view was requested, trigger a warning.
    if not copy and np.any(getattr(array, "mask", False)):
        warnings.warn("Copying is required to respect the mask. Returning copy. Set 'copy=True' to hide this message.")
        copy = True

    # If array is of type integer and has a mask, it needs to be converted to float (to assign nans)
    if np.any(getattr(array, "mask", False)) and np.issubdtype(array.dtype, np.integer):  # type: ignore
        array = array.astype(np.float32)  # type: ignore

    # Convert into a regular ndarray (a view or copy depending on the 'copy' argument)
    array_data = np.array(array).squeeze() if copy else np.asarray(array).squeeze()

    # Get the mask of invalid pixels and set nans if it is occupied.
    invalid_mask = get_mask(array)
    if np.any(invalid_mask):
        array_data[invalid_mask] = np.nan

    return array_data, invalid_mask


def get_valid_extent(array: np.ndarray | np.ma.masked_array) -> tuple[int, ...]:
    """
    Return (rowmin, rowmax, colmin, colmax), the first/last row/column of array with valid pixels
    """
    if not array.dtype == "bool":
        valid_mask = ~get_mask(array)
    else:
        valid_mask = array
    cols_nonzero = np.where(np.count_nonzero(valid_mask, axis=0) > 0)[0]
    rows_nonzero = np.where(np.count_nonzero(valid_mask, axis=1) > 0)[0]
    return rows_nonzero[0], rows_nonzero[-1], cols_nonzero[0], cols_nonzero[-1]


def load_multiple_rasters(
    raster_paths: list[str], crop: bool = True, ref_grid: int | None = None, **kwargs: Any
) -> list[RasterType]:
    """
    Function to load multiple rasters at once in a memory efficient way.
    First load metadata only.
    Optionally, crop all rasters to their intersection (default).
    Optionally, reproject all rasters to the grid of one raster set as reference (after optional crop).
    Otherwise, simply load the full rasters.

    :param raster_paths: List of paths to the rasters to be loaded
    :param crop: if set to True, will only load rasters in the area they intersect
    :param ref_grid: If set to an integer value, the raster with that index will be considered as the reference
    and all other rasters will be reprojected on the same grid (after optional crop)
    :param kwargs: optional arguments to be passed to Raster.reproject, e.g. the resampling method

    :returns: a list of loaded Raster instances
    """
    # If ref_grid is provided, need to reproject
    if isinstance(ref_grid, int):
        reproject = True
    # if no ref_grid provided, still need a reference CRS, use first by default
    elif ref_grid is None:
        ref_grid = 0
        reproject = False
    else:
        raise ValueError("`ref_grid` must be None or an integer")

    # Need to define a reference CRS for calculating intersection
    ref_crs = gu.Raster(raster_paths[ref_grid], load_data=False).crs

    # First load all rasters metadata
    output_rst = []
    bounds = []
    for path in raster_paths:
        # Initialize raster
        rst = gu.Raster(path, load_data=False)
        output_rst.append(rst)

        # Get bound in reference CRS
        bound = rst.get_bounds_projected(ref_crs)
        bounds.append(bound)

    # Second get the intersection of all raster bounds
    intersection = gu.projtools.merge_bounds(bounds, "intersection")

    # Optionally, crop the rasters
    if crop:
        # Check that intersection is not void
        if intersection == ():
            warnings.warn("Intersection is void, returning unloaded rasters.")
            return output_rst

        for rst in output_rst:
            # Calculate bounds in rst's CRS
            # rasterio's default for densify_pts is too low for very large images, set a default of 5000
            new_bounds = rio.warp.transform_bounds(
                ref_crs, rst.crs, intersection[0], intersection[1], intersection[2], intersection[3], densify_pts=5000
            )
            # Ensure bounds align with the original ones, to avoid resampling at this stage
            new_bounds = gu.projtools.align_bounds(rst.transform, new_bounds)
            rst.crop(new_bounds, mode="match_pixel")

    # Optionally, reproject all rasters to the reference grid
    if reproject:

        ref_rst = output_rst[ref_grid]

        # Set output bounds - intersection if crop is True, otherwise use that of ref_grid
        if crop:
            # make sure new bounds align with reference's bounds (to avoid resampling ref)
            new_bounds = intersection
            new_bounds = gu.projtools.align_bounds(ref_rst.transform, intersection)
        else:
            new_bounds = ref_rst.bounds

        # Reproject all rasters
        for index, rst in enumerate(output_rst):
            out_rst = rst.reproject(
                dst_crs=ref_rst.crs, dst_bounds=new_bounds, dst_res=ref_rst.res, silent=True, **kwargs
            )
            if not out_rst.is_loaded:
                out_rst.load()
            output_rst[index] = out_rst

    # if no crop or reproject option, simply load the rasters
    if (not crop) & (not reproject):
        for rst in output_rst:
            rst.load()

    return output_rst


def merge_bounding_boxes(bounds: list[rio.coords.BoundingBox], resolution: float) -> rio.coords.BoundingBox:
    max_bounds = dict(zip(["left", "right", "top", "bottom"], [np.nan] * 4))
    for bound in bounds:
        for key in "right", "top":
            max_bounds[key] = np.nanmax([max_bounds[key], bound.__getattribute__(key)])
        for key in "bottom", "left":
            max_bounds[key] = np.nanmin([max_bounds[key], bound.__getattribute__(key)])

    # Make sure that extent is a multiple of resolution
    for key1, key2 in zip(("left", "bottom"), ("right", "top")):
        modulo = (max_bounds[key2] - max_bounds[key1]) % resolution
        max_bounds[key2] += modulo

    return rio.coords.BoundingBox(**max_bounds)


def stack_rasters(
    rasters: list[RasterType],
    reference: int | gu.Raster = 0,
    resampling_method: str | rio.warp.Resampling = "bilinear",
    use_ref_bounds: bool = False,
    diff: bool = False,
    progress: bool = True,
) -> gu.Raster:
    """
    Stack a list of rasters into a common grid as a 3D np array with nodata set to Nan.

    If use_ref_bounds is True, output will have the shape (N, height, width) where N is len(rasters) and \
height and width is equal to reference's shape.
    If use_ref_bounds is False, output will have the shape (N, height2, width2) where N is len(rasters) and \
height2 and width2 are set based on reference's resolution and the maximum extent of all rasters.

    Use diff=True to return directly the difference to the reference raster.

    Note that currently all rasters will be loaded once in memory. However, if rasters data is not loaded prior to \
    merge_rasters it will be loaded for reprojection and deleted, therefore avoiding duplication and \
    optimizing memory usage.

    :param rasters: A list of geoutils Raster objects to be stacked.
    :param reference: The reference index, in case the reference is to be stacked, or a separate Raster object \
 in case the reference should not be stacked. Defaults to the first raster in the list.
    :param resampling_method: The resampling method for the raster reprojections.
    :param use_ref_bounds: If True, will use reference bounds, otherwise will use maximum bounds of all rasters.
    :param diff: If True, will return the difference to the reference raster.
    :param progress: If True, will display a progress bar. Default is True.

    :returns: The stacked raster with the same parameters (optionally bounds) as the reference.
    """
    # Check resampling method
    if isinstance(resampling_method, str):
        resampling_method = resampling_method_from_str(resampling_method)

    # Select reference raster
    if isinstance(reference, int):
        reference_raster = rasters[reference]
    elif isinstance(reference, gu.Raster):
        reference_raster = reference
    else:
        raise ValueError("reference should be either an integer or geoutils.Raster object")

    # Set output bounds
    if use_ref_bounds:
        dst_bounds = reference_raster.bounds
    else:
        dst_bounds = merge_bounding_boxes([raster.bounds for raster in rasters], resolution=reference_raster.res[0])

    # Make a data list and add all of the reprojected rasters into it.
    data: list[np.ndarray] = []

    for raster in tqdm(rasters, disable=not progress):

        # Check that data is loaded, otherwise temporarily load it
        if not raster.is_loaded:
            raster.load()
            raster.is_loaded = False

        nodata = reference_raster.nodata or gu.georaster.raster._default_nodata(reference_raster.data.dtype)
        # Reproject to reference grid
        reprojected_raster = raster.reproject(
            dst_bounds=dst_bounds,
            dst_res=reference_raster.res,
            dst_crs=reference_raster.crs,
            dtype=reference_raster.data.dtype,
            dst_nodata=reference_raster.nodata,
            silent=True,
        )
        reprojected_raster.set_nodata(nodata)

        # Optionally calculate difference
        if diff:
            diff_to_ref = (reference_raster.data - reprojected_raster.data).squeeze()
            diff_to_ref, _ = get_array_and_mask(diff_to_ref)
            data.append(diff_to_ref)
        else:
            # img_data, _ = get_array_and_mask(reprojected_raster.data.squeeze())
            data.append(reprojected_raster.data.squeeze())

        # Remove unloaded rasters
        if not raster.is_loaded:
            raster._data = None

    # Convert to masked array
    data = np.ma.asarray(data)
    if reference_raster.nodata is not None:
        nodata = reference_raster.nodata
    else:
        nodata = _default_nodata(data.dtype)
    data[np.isnan(data)] = nodata

    # Save as gu.Raster - needed as some child classes may not accept multiple bands
    r = gu.Raster.from_array(
        data=data,
        transform=rio.transform.from_bounds(*dst_bounds, width=data[0].shape[1], height=data[0].shape[0]),
        crs=reference_raster.crs,
        nodata=nodata,
    )

    return r


def merge_rasters(
    rasters: list[RasterType],
    reference: int | Raster = 0,
    merge_algorithm: Callable | list[Callable] = np.nanmean,  # type: ignore
    resampling_method: str | rio.warp.Resampling = "bilinear",
    use_ref_bounds: bool = False,
    progress: bool = True,
) -> RasterType:
    """
    Merge a list of rasters into one larger raster.

    Reprojects the rasters to the reference raster CRS and resolution.
    Note that currently all rasters will be loaded once in memory. However, if rasters data is not loaded prior to \
    merge_rasters it will be loaded for reprojection and deleted, therefore avoiding duplication and \
    optimizing memory usage.

    :param rasters: A list of geoutils Raster objects to be merged.
    :param reference: The reference index, in case the reference is to be merged, or a separate Raster object \
 in case the reference should not be merged. Defaults to the first raster in the list.
    :param merge_algorithm: The algorithm, or list of algorithms, to merge the rasters with. Defaults to the mean.\
If several algorithms are provided, each result is returned as a separate band.
    :param resampling_method: The resampling method for the raster reprojections.
    :param use_ref_bounds: If True, will use reference bounds, otherwise will use maximum bounds of all rasters.
    :param progress: If True, will display a progress bar. Default is True.

    :returns: The merged raster with the same parameters (excl. bounds) as the reference.
    """
    # Make sure merge_algorithm is a list
    if not isinstance(merge_algorithm, (list, tuple)):
        merge_algorithm = [
            merge_algorithm,
        ]

    # Try to run the merge_algorithm with an arbitrary list. Raise an error if the algorithm is incompatible.
    for algo in merge_algorithm:
        try:
            algo([1, 2])
        except TypeError as exception:
            raise TypeError(f"merge_algorithm must be able to take a list as its first argument.\n\n{exception}")

    # Select reference raster
    if isinstance(reference, int):
        reference_raster = rasters[reference]
    elif isinstance(reference, gu.Raster):
        reference_raster = reference
    else:
        raise ValueError("reference should be either an integer or geoutils.Raster object")

    # Reproject and stack all rasters
    raster_stack = stack_rasters(
        rasters,
        reference=reference,
        resampling_method=resampling_method,
        use_ref_bounds=use_ref_bounds,
        progress=progress,
    )

    # Try to use the keyword axis=0 for the merging algorithm (if it's a numpy ufunc).
    merged_data = []
    for algo in merge_algorithm:
        try:
            merged_data.append(algo(raster_stack.data, axis=0))
        # If that doesn't work, use the slower np.apply_along_axis approach.
        except TypeError as exception:
            if "'axis' is an invalid keyword" not in str(exception):
                raise exception
            merged_data.append(np.apply_along_axis(algo, axis=0, arr=raster_stack.data))

    # Convert to masked array, and set all Nans to nodata
    merged_data = np.ma.asarray(merged_data)
    if reference_raster.nodata is not None:
        nodata = reference_raster.nodata
    else:
        nodata = _default_nodata(merged_data.dtype)
    merged_data[np.isnan(merged_data)] = nodata

    # Save as gu.Raster
    merged_raster = reference_raster.from_array(
        data=np.reshape(merged_data, (len(merged_data),) + merged_data[0].shape),
        transform=rio.transform.from_bounds(
            *raster_stack.bounds, width=merged_data[0].shape[1], height=merged_data[0].shape[0]
        ),
        crs=reference_raster.crs,
        nodata=nodata,
    )

    return merged_raster


def _get_closest_rectangle(size: int) -> tuple[int, int]:
    """
    Given a 1D array size, return a rectangular shape that is closest to a cube which the size fits in.

    If 'size' does not have an integer root, a rectangle is returned that is slightly larger than 'size'.

    :examples:
        >>> _get_closest_rectangle(4)  # size will be 4
        (2, 2)
        >>> _get_closest_rectangle(9)  # size will be 9
        (3, 3)
        >>> _get_closest_rectangle(3)  # size will be 4; needs padding afterward.
        (2, 2)
        >>> _get_closest_rectangle(55) # size will be 56; needs padding afterward.
        (7, 8)
        >>> _get_closest_rectangle(24)  # size will be 25; needs padding afterward
        (5, 5)
        >>> _get_closest_rectangle(85620)  # size will be 85849; needs padding afterward
        (293, 293)
        >>> _get_closest_rectangle(52011)  # size will be 52212; needs padding afterward
        (228, 229)
    """
    close_cube = int(np.sqrt(size))

    # If size has an integer root, return the respective cube.
    if close_cube**2 == size:
        return (close_cube, close_cube)

    # One of these rectangles/cubes will cover all cells, so return the first that does.
    potential_rectangles = [(close_cube, close_cube + 1), (close_cube + 1, close_cube + 1)]

    for rectangle in potential_rectangles:
        if np.prod(rectangle) >= size:
            return rectangle

    raise NotImplementedError(f"Function criteria not met for rectangle of size: {size}")


def subdivide_array(shape: tuple[int, ...], count: int) -> np.ndarray:
    """
    Create indices for subdivison of an array in a number of blocks.

    If 'count' is divisible by the product of 'shape', the amount of cells in each block will be equal.
    If 'count' is not divisible, the amount of cells in each block will be very close to equal.

    :param shape: The shape of a array to be subdivided.
    :param count: The amount of subdivisions to make.

    :examples:
        >>> subdivide_array((4, 4), 4)
        array([[0, 0, 1, 1],
               [0, 0, 1, 1],
               [2, 2, 3, 3],
               [2, 2, 3, 3]])

        >>> subdivide_array((6, 4), 4)
        array([[0, 0, 1, 1],
               [0, 0, 1, 1],
               [0, 0, 1, 1],
               [2, 2, 3, 3],
               [2, 2, 3, 3],
               [2, 2, 3, 3]])

        >>> subdivide_array((5, 4), 3)
        array([[0, 0, 0, 0],
               [0, 0, 0, 0],
               [1, 1, 2, 2],
               [1, 1, 2, 2],
               [1, 1, 2, 2]])

    :raises ValueError: If the 'shape' size (`np.prod(shape)`) is smallern than 'count'
                        If the shape is not a 2D shape.

    :returns: An array of shape 'shape' with 'count' unique indices.
    """
    try:
        import skimage.transform
    except ImportError:
        raise ImportError("Missing optional dependency, skimage.transform, required by this function.")

    if count > np.prod(shape):
        raise ValueError(f"Shape '{shape}' size ({np.prod(shape)}) is smaller than 'count' ({count}).")

    if len(shape) != 2:
        raise ValueError(f"Expected a 2D shape, got {len(shape)}D shape: {shape}")

    # Generate a small grid of indices, with the same unique count as 'count'
    rect = _get_closest_rectangle(count)
    small_indices = np.pad(np.arange(count), np.prod(rect) - count, mode="edge")[: np.prod(rect)].reshape(rect)

    # Upscale the grid to fit the output shape using nearest neighbour scaling.
    indices = skimage.transform.resize(small_indices, shape, order=0, preserve_range=True).astype(int)

    return indices.reshape(shape)


def get_xy_rotated(raster: Raster, along_track_angle: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotate x, y axes of image to get along- and cross-track distances.
    :param raster: raster to get x,y positions from.
    :param along_track_angle: angle by which to rotate axes (degrees)
    :returns xxr, yyr: arrays corresponding to along (x) and cross (y) track distances.
    """

    myang = np.deg2rad(along_track_angle)

    # get grid coordinates
    xx, yy = raster.coords(grid=True)
    xx -= np.min(xx)
    yy -= np.min(yy)

    # get rotated coordinates

    # for along-track
    xxr = np.multiply(xx, np.cos(myang)) + np.multiply(-1 * yy, np.sin(along_track_angle))
    # for cross-track
    yyr = np.multiply(xx, np.sin(myang)) + np.multiply(yy, np.cos(along_track_angle))

    # re-initialize coordinate at zero
    xxr -= np.nanmin(xxr)
    yyr -= np.nanmin(yyr)

    return xxr, yyr


def subsample_raster(
    array: np.ndarray | np.ma.masked_array,
    subsample: float | int,
    return_indices: bool = False,
    random_state: None | np.random.RandomState | np.random.Generator | int = None,
) -> np.ndarray:
    """
    Randomly subsample a 1D or 2D array by a subsampling factor, taking only non NaN/masked values.

    :param subsample: If <= 1, will be considered a fraction of valid pixels to extract.
    If > 1 will be considered the number of pixels to extract.
    :param return_indices: If set to True, will return the extracted indices only.
    :param random_state: Random state, or seed number to use for random calculations (for testing)

    :returns: The subsampled array (1D) or the indices to extract (same shape as input array)
    """
    # Define state for random subsampling (to fix results during testing)
    if random_state is None:
        rnd = np.random.default_rng()
    elif isinstance(random_state, (np.random.RandomState, np.random.Generator)):
        rnd = random_state
    else:
        rnd = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(random_state)))

    # Get number of points to extract
    if (subsample <= 1) & (subsample > 0):
        npoints = int(subsample * np.size(array))
    elif subsample > 1:
        npoints = int(subsample)
    else:
        raise ValueError("`subsample` must be > 0")

    # Remove invalid values and flatten array
    mask = get_mask(array)  # -> need to remove .squeeze in get_mask
    valids = np.argwhere(~mask.flatten()).squeeze()

    # Checks that array and npoints are correct
    assert np.ndim(valids) == 1, "Something is wrong with array dimension, check input data and shape"
    if npoints > np.size(valids):
        npoints = np.size(valids)

    # Randomly extract npoints without replacement
    indices = rnd.choice(valids, npoints, replace=False)
    unraveled_indices = np.unravel_index(indices, array.shape)

    if return_indices:
        return unraveled_indices

    else:
        return array[unraveled_indices]
