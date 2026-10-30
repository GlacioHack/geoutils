from __future__ import annotations

import builtins
import os
import re
import subprocess
import sys
import warnings
from configparser import ConfigParser
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import geopandas as gpd
import pytest
import yaml  # type: ignore
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

import geoutils
from geoutils._misc import (
    _trim_process_memory,
    copy_doc,
    deprecate,
    diff_environment_yml,
    import_optional,
)


class TestMisc:
    def test_environment_files(self) -> None:
        """Check that environment yml files are properly written: all dependencies of env are also in dev-env"""

        fn_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "environment.yml"))
        fn_devenv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dev-environment.yml"))

        # Load the yml as dictionaries
        yaml_env = yaml.safe_load(open(fn_env))
        yaml_devenv = yaml.safe_load(open(fn_devenv))

        # Extract the dependencies values
        conda_dep_env = yaml_env["dependencies"]
        conda_dep_devenv = yaml_devenv["dependencies"]

        # Check if there is any pip dependency, if yes pop it from the end of the list
        if isinstance(conda_dep_devenv[-1], dict):
            pip_dep_devenv = conda_dep_devenv.pop()

            # Check if there is a pip dependency in the normal env as well, if yes pop it also
            if isinstance(conda_dep_env[-1], dict):
                pip_dep_env = conda_dep_env.pop()

                # The diff below computes the dependencies that are in env but not in dev-env
                # It should be empty, otherwise we raise an error
                diff_pip_check = list(set(pip_dep_env) - set(pip_dep_devenv))
                assert len(diff_pip_check) == 0

        # We do the same for the conda dependency, first a sanity check that everything that is in env is also in dev-ev
        # Compare package names because the development environment can require newer versions
        conda_names_env = {re.split(r"[<>=!]", dependency)[0] for dependency in conda_dep_env}
        conda_names_devenv = {re.split(r"[<>=!]", dependency)[0] for dependency in conda_dep_devenv}
        diff_conda_check = list(conda_names_env - conda_names_devenv)
        assert len(diff_conda_check) == 0

    def test_dev_environment__match(self) -> None:
        """Checks that setup.cfg dev dependencies are declared in dev-environment.yml for conda."""

        # Expand the dev extra, including its referenced extras
        project_root = Path(__file__).resolve().parents[1]
        config = ConfigParser()
        config.read(project_root / "setup.cfg")
        requirements = config["options.extras_require"]["dev"].splitlines()
        required_names = {canonicalize_name(Requirement(line).name) for line in requirements if line.strip()}

        # Read Conda and pip declarations
        with (project_root / "dev-environment.yml").open() as stream:
            dependencies = yaml.safe_load(stream)["dependencies"]
        declared_names = set()
        for dependency in dependencies:
            if isinstance(dependency, str):
                name = re.split(r"[<>=!]", dependency, maxsplit=1)[0].strip()
                declared_names.add(canonicalize_name(name))
            else:
                for requirement in dependency["pip"]:
                    # Remove pip editable line
                    if requirement.startswith("-e "):
                        continue
                    declared_names.add(canonicalize_name(Requirement(requirement).name))

        # Compare names
        missing_names = sorted(required_names - declared_names)
        assert not missing_names, f"Development dependencies missing from dev-environment.yml: {missing_names}"

    @pytest.mark.parametrize("deprecation_increment", [-1, 0, 1, None])
    @pytest.mark.parametrize("details", [None, "It was completely useless!", "dunnowhy"])
    def test_deprecate(self, deprecation_increment: int | None, details: str | None) -> None:
        """
        Test the deprecation warnings/errors.

        If the removal_version is larger than the current, it should warn.
        If the removal_version is smaller or equal, it should raise an error.

        :param deprecation_increment: The version number relative to the current version.
        :param details: An optional explanation for the description.
        """
        warnings.simplefilter("error")

        current_version = Version(Version(geoutils.__version__).base_version)

        # Set the removal version to be the current version plus the increment (e.g. 0.0.5 + 1 -> 0.0.6)
        if deprecation_increment is not None:
            # If the micro version is already 0 and it is a decrement, decrement minor version instead
            if current_version.micro == 0 and deprecation_increment == -1:
                removal_version_tuple = (current_version.major, current_version.minor + deprecation_increment, 0)
            # Otherwise, increment micro version
            else:
                removal_version_tuple = (
                    current_version.major,
                    current_version.minor,
                    current_version.micro + deprecation_increment,
                )

            # Convert to version
            removal_version = Version(".".join([str(v) for v in removal_version_tuple]))
        else:
            removal_version = None

        # Define a function with no use that is marked as deprecated.
        @deprecate(removal_version, details=details)
        def useless_func() -> int:
            return 1

        # Example of why Version needs to be used below
        assert not "0.0.10" > "0.0.8"
        assert Version("0.0.10") > Version("0.0.8")

        # If True, a warning is expected. If False, a ValueError is expected.
        should_warn = removal_version is None or removal_version > current_version

        # Add the expected text depending on the parametrization.
        text = (
            "Call to deprecated function 'useless_func'."
            if should_warn
            else f"Deprecated function 'useless_func' was removed in {removal_version}."
        )

        if details is not None:
            text += " " + details.strip().capitalize()

            if not any(text.endswith(c) for c in ".!?"):
                text += "."

        if should_warn and removal_version is not None:
            text += f" This functionality will be removed in version {removal_version}."
        elif not should_warn:
            text += f" Current version: {current_version}."

        # Expect either a warning or an exception with the right text.
        if should_warn:
            with pytest.warns(DeprecationWarning, match="^" + text + "$"):
                useless_func()
        else:
            with pytest.raises(ValueError, match=re.escape(text)):
                useless_func()

    def test_diff_environment_yml(self, capsys) -> None:  # type: ignore
        # Test with synthetic environment
        env = {"dependencies": ["python==3.9", "numpy", "pandas"]}
        devenv = {"dependencies": ["python==3.9", "numpy", "pandas", "opencv"]}

        # This should print the difference between the two
        diff_environment_yml(env, devenv, input_dict=True, print_dep="conda")

        # Capture the stdout and check it is indeed the right diff
        captured = capsys.readouterr().out
        assert captured == "opencv\n"

        # This should print the difference including pip
        diff_environment_yml(env, devenv, input_dict=True, print_dep="both")

        captured = capsys.readouterr().out
        assert captured == "opencv\nNone\n"

        env2 = {"dependencies": ["python==3.9", "numpy", "pandas"]}
        devenv2 = {"dependencies": ["python==3.9", "numpy", "pandas", "opencv", {"pip": ["geoutils", "-e ./"]}]}

        # The diff function should not account for -e ./ that is the local install for developers
        diff_environment_yml(env2, devenv2, input_dict=True, print_dep="both")
        captured = capsys.readouterr().out

        assert captured == "opencv\ngeoutils\n"

        # This should print only pip
        diff_environment_yml(env2, devenv2, input_dict=True, print_dep="pip")
        captured = capsys.readouterr().out

        assert captured == "geoutils\n"

        # This should raise an error because print_dep is not well defined
        with pytest.raises(ValueError, match='The argument "print_dep" can only be "conda", "pip" or "both".'):
            diff_environment_yml(env2, devenv2, input_dict=True, print_dep="lol")

        # When the dependencies are not defined in dev-env but in env, it should raise an error
        # For normal dependencies
        env3 = {"dependencies": ["python==3.9", "numpy", "pandas", "lol"]}
        devenv3 = {"dependencies": ["python==3.9", "numpy", "pandas", "opencv", {"pip": ["geoutils"]}]}
        with pytest.raises(ValueError, match="The following dependencies are listed in env but not dev-env: lol"):
            diff_environment_yml(env3, devenv3, input_dict=True, print_dep="pip")

        # For pip dependencies
        env4 = {"dependencies": ["python==3.9", "numpy", "pandas", {"pip": ["lol"]}]}
        devenv4 = {"dependencies": ["python==3.9", "numpy", "pandas", "opencv", {"pip": ["geoutils"]}]}
        with pytest.raises(ValueError, match="The following pip dependencies are listed in env but not dev-env: lol"):
            diff_environment_yml(env4, devenv4, input_dict=True, print_dep="pip")

    def test_copydoc_geopandas(self) -> None:

        # This function exist in geopandas, so should not fail and copy the doc with "Vector" in it
        @copy_doc(gpd.GeoSeries, "Vector")
        def union() -> int:
            return 1

        assert "Vector" in union.__doc__

        # This function does not exist, so should not fail either (to ensure GeoUtils package import), but should
        # have a description stating that it does not exist.
        @copy_doc(gpd.GeoSeries, "Vector")
        def doesnotexist() -> int:
            return 1

        assert doesnotexist.__doc__ == "This function documentation does not exist in GeoPandas (likely deprecated)."

    def test_trim_process_memory__missing_process_library(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks that memory trimming is skipped when the platform does not support it."""

        import ctypes

        # Simulate Windows, where CDLL(None) raises TypeError instead of exposing the C process library
        def reject_null_library(name: str | None) -> None:
            assert name is None
            raise TypeError("argument of type 'NoneType' is not iterable")

        monkeypatch.setattr(ctypes, "CDLL", reject_null_library)

        # Check the worker stays usable
        _trim_process_memory()


class TestOptionalImports:
    """Test module for optional dependency errors and GPyTorch import warnings."""

    @pytest.mark.parametrize(
        "dependency, warning_module, torch_warning, filtered",
        [
            ("gpytorch", "torch.jit._script", True, True),
            ("gstools", "torch.jit._script", True, False),
            ("gpytorch", "another_module", True, False),
            ("gpytorch", "torch.jit._script", False, False),
        ],
    )
    def test_import_optional__scoped_torch_warning(
        self,
        dependency: str,
        warning_module: str,
        torch_warning: bool,
        filtered: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Checks that only the upstream TorchScript warning is filtered during a GPyTorch import."""

        message = (
            "`torch.jit.script` is not supported in Python 3.14+ and may break. "
            "Please switch to `torch.compile` or `torch.export`."
            if torch_warning
            else "Another upstream warning"
        )
        imported = object()
        original_import = builtins.__import__

        def import_with_warning(name: str, *args: Any, **kwargs: Any) -> Any:
            """Simulate an upstream import warning without loading optional packages."""

            if name == dependency:
                warnings.warn_explicit(message, FutureWarning, filename="upstream.py", lineno=1, module=warning_module)
                return imported
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", import_with_warning)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            original_filters = warnings.filters.copy()
            assert import_optional(dependency) is imported
            assert warnings.filters == original_filters
            assert len(caught) == int(not filtered)

            # Check warnings after import
            warnings.warn_explicit(message, FutureWarning, filename="upstream.py", lineno=2, module=warning_module)
            assert len(caught) == 1 + int(not filtered)

    def test_import_optional__error_missing_dependency(self) -> None:
        """Checks an error is raised with installation guidance for a missing optional dependency."""

        with pytest.raises(ImportError, match="Optional dependency '_geoutils_missing_dependency_' required"):
            import_optional("_geoutils_missing_dependency_")

    @pytest.mark.skipif(find_spec("gpytorch") is None, reason="Requires GPyTorch")
    def test_import_optional__gpytorch_sampling(self) -> None:
        """Checks that GPyTorch imports and samples reproducibly with the TorchScript warning treated as an error."""

        # Avoid cached optional imports
        code = r"""
import warnings
import numpy as np
import geoutils as gu
from geoutils._misc import import_optional

warnings.filterwarnings('error', message=r'.*torch\.jit\.script.*Python 3\.14', category=FutureWarning)
import_optional('gpytorch')
model = gu.Variogram.from_model('gaussian', effective_range=2, partial_sill=1)
errors = gu.ErrorStructure([gu.ErrorComponent('spatial', 1, model)])
coordinates = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
first = gu.uncertainty.random_field(errors, coordinates=coordinates, backend='gpytorch', random_state=42)
second = gu.uncertainty.random_field(errors, coordinates=coordinates, backend='gpytorch', random_state=42)
assert first.shape == (4,)
assert np.isfinite(first).all()
np.testing.assert_array_equal(first, second)
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
