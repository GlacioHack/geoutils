"""Miscellaneous functions, mainly for testing."""
from __future__ import annotations

import functools
import warnings

try:
    import yaml  # type: ignore

    _has_yaml = True
except ImportError:
    _has_yaml = False

import rasterio as rio
from packaging.version import Version

import geoutils


def deprecate(removal_version: str | None = None, details: str | None = None):  # type: ignore
    """
    Trigger a DeprecationWarning for the decorated function.

    :param func: The function to be deprecated.
    :param removal_version: Optional. The version at which this will be removed.
                            If this version is reached, a ValueError is raised.
    :param details: Optional. A description for why the function was deprecated.

    :triggers DeprecationWarning: For any call to the function.

    :raises ValueError: If 'removal_version' was given and the current version is equal or higher.

    :returns: The decorator to decorate the function.
    """

    def deprecator_func(func):  # type: ignore
        @functools.wraps(func)
        def new_func(*args, **kwargs):  # type: ignore
            # True if it should warn, False if it should raise an error
            should_warn = removal_version is None or Version(removal_version) > Version(geoutils.version.version)

            # Add text depending on the given arguments and 'should_warn'.
            text = (
                f"Call to deprecated function '{func.__name__}'."
                if should_warn
                else f"Deprecated function '{func.__name__}' was removed in {removal_version}."
            )

            # Add the details explanation if it was given, and make sure the sentence is ended.
            if details is not None:
                details_frm = details.strip()
                if details_frm[0].islower():
                    details_frm = details_frm[0].upper() + details_frm[1:]

                text += " " + details_frm

                if not any(text.endswith(c) for c in ".!?"):
                    text += "."

            if should_warn and removal_version is not None:
                text += f" This functionality will be removed in version {removal_version}."
            elif not should_warn:
                text += f" Current version: {geoutils.version.version}."

            if should_warn:
                warnings.warn(text, category=DeprecationWarning, stacklevel=2)
            else:
                raise ValueError(text)

            return func(*args, **kwargs)

        return new_func

    return deprecator_func


def resampling_method_from_str(method_str: str) -> rio.warp.Resampling:
    """Get a rasterio resampling method from a string representation, e.g. "cubic_spline"."""
    # Try to match the string version of the resampling method with a rio Resampling enum name
    for method in rio.warp.Resampling:
        if str(method).replace("Resampling.", "") == method_str:
            resampling_method = method
            break
    # If no match was found, raise an error.
    else:
        raise ValueError(
            f"'{method_str}' is not a valid rasterio.warp.Resampling method. "
            f"Valid methods: {[str(method).replace('Resampling.', '') for method in rio.warp.Resampling]}"
        )
    return resampling_method


def diff_environment_yml(fn_env: str, fn_devenv: str, print_dep: str = "both") -> None:
    """
    Compute the difference between environment.yml and dev-environment.yml for setup of continuous integration,
    while checking that all the dependencies listed in environment.yml are also in dev-environment.yml

    :param fn_env: Filename path to environment.yml
    :param fn_devenv: Filename path to dev-environment.yml
    :param print_dep: Whether to print conda differences "conda", pip differences "pip" or both.
    """

    if not _has_yaml:
        raise ValueError("Test dependency needed. Install 'pyyaml'")

    # Load the yml as dictionaries
    yaml_env = yaml.safe_load(open(fn_env))
    yaml_devenv = yaml.safe_load(open(fn_devenv))

    # Extract the dependencies values
    conda_dep_env = yaml_env["dependencies"]
    conda_dep_devenv = yaml_devenv["dependencies"]

    # Check if there is any pip dependency, if yes pop it from the end of the list
    if isinstance(conda_dep_devenv[-1], dict):
        pip_dep_devenv = conda_dep_devenv.pop()["pip"]

        # Check if there is a pip dependency in the normal env as well, if yes pop it also
        if isinstance(conda_dep_env[-1], dict):
            pip_dep_env = conda_dep_env.pop()["pip"]

            # The diff below computes the dependencies that are in env but not in dev-env
            # It should be empty, otherwise we raise an error
            diff_pip_check = list(set(pip_dep_env) - set(pip_dep_devenv))
            if len(diff_pip_check) != 0:
                raise ValueError(
                    "The following pip dependencies are listed in env but not dev-env: " + ",".join(diff_pip_check)
                )

            # The diff below computes the dependencies that are in dev-env but not in env, to add during CI
            diff_pip_dep = list(set(pip_dep_devenv) - set(pip_dep_env))

        # If there is no pip dependency in env, all the ones of dev-env need to be added during CI
        else:
            diff_pip_dep = list(pip_dep_devenv["pip"])

    # If there is no pip dependency, we ignore this step
    else:
        diff_pip_dep = []

    # If the diff is empty for pip, return a string "None" to read easily in bash
    if len(diff_pip_dep) == 0:
        diff_pip_dep = ["None"]

    # We do the same for the conda dependency, first a sanity check that everything that is in env is also in dev-ev
    diff_conda_check = list(set(conda_dep_env) - set(conda_dep_devenv))
    if len(diff_conda_check) != 0:
        raise ValueError("The following dependencies are listed in env but not dev-env: " + ",".join(diff_conda_check))

    # Then the difference to add during CI
    diff_conda_dep = list(set(conda_dep_devenv) - set(conda_dep_env))

    # Join the lists
    joined_list_conda_dep = " ".join(diff_conda_dep)
    joined_list_pip_dep = " ".join(diff_pip_dep)

    # Print to be captured in bash
    if print_dep == "both":
        print(joined_list_conda_dep)
        print(joined_list_pip_dep)
    elif print_dep == "conda":
        print(joined_list_conda_dep)
    elif print_dep == "pip":
        print(joined_list_pip_dep)
    else:
        raise ValueError('The argument "print_dep" can only be "conda", "pip" or "both".')
