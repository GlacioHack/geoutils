"""Test the license header checker used by pre-commit."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / ".github" / "scripts"
_CHECKER = _SCRIPTS / "check_license_header.py"
_BODY = (_SCRIPTS / "license-header.txt").read_text(encoding="utf-8").splitlines()[1:]
_GEOUTILS = "# Copyright (c) 2021 GeoUtils developers"
_CNES = "# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES)"
_CONTENT = "\n".join([_GEOUTILS, *_BODY]) + "\n\nvalue = 1\n"


class TestLicenseHeader:
    """Test module for accepted copyright lines and files left unchanged by the license header checker."""

    @pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
    @pytest.mark.parametrize(
        "copyright_lines",
        [
            pytest.param([_GEOUTILS], id="geoutils"),
            pytest.param([_CNES], id="cnes"),
            pytest.param([_GEOUTILS, _CNES], id="both"),
            pytest.param([_CNES, _GEOUTILS], id="reversed"),
        ],
    )
    def test_main__accepted_headers(self, tmp_path: Path, copyright_lines: list[str], newline: str) -> None:
        """Checks that the checker accepts existing holders and different orders without changing the file."""

        # Preserve historical years and line endings
        filename = tmp_path / "example.py"
        content = newline.join([*copyright_lines, *_BODY, "", "value = 1", ""]).encode("utf-8")
        filename.write_bytes(content)

        result = subprocess.run([sys.executable, str(_CHECKER), str(filename)], capture_output=True, text=True)

        assert result.returncode == 0, result.stdout + result.stderr
        assert filename.read_bytes() == content


class TestLicenseHeaderErrors:
    """Test module for malformed headers and file errors reported together by the license header checker."""

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("value = 1\n", id="missing-header"),
            pytest.param("\n".join(_BODY) + "\n\nvalue = 1\n", id="missing-copyright"),
            pytest.param(_GEOUTILS + "\n\nvalue = 1\n", id="missing-body"),
            pytest.param(_CONTENT.replace("Apache License, Version 2.0", "MIT License"), id="license-text"),
            pytest.param(_CONTENT.split("# Unless required", maxsplit=1)[0], id="truncated"),
            pytest.param(_CONTENT.replace("GeoUtils developers", "Unknown developers"), id="wrong-owner"),
            pytest.param(_CONTENT.replace("GeoUtils developers", "xDEM developers"), id="xdem-owner"),
            pytest.param(_CONTENT.replace("2021", "21"), id="invalid-year"),
            pytest.param("\n" + _CONTENT, id="leading-text"),
            pytest.param(_CONTENT.replace("# http://www.apache.org", "#     http://www.apache.org"), id="whitespace"),
            pytest.param(_CONTENT.replace("GeoUtils project", "Other project"), id="project-name"),
        ],
    )
    def test_main__error_invalid_header(self, tmp_path: Path, content: str) -> None:
        """Checks an error is raised for an invalid header and the checker leaves the file unchanged."""

        filename = tmp_path / "invalid.py"
        original = content.encode("utf-8")
        filename.write_bytes(original)

        result = subprocess.run([sys.executable, str(_CHECKER), str(filename)], capture_output=True, text=True)

        assert result.returncode == 1
        assert str(filename) in result.stdout + result.stderr
        assert filename.read_bytes() == original

    def test_main__error_reports_all_files(self, tmp_path: Path) -> None:
        """Checks an error is raised for every invalid or unreadable file while valid files remain unchanged."""

        # A directory cannot be read as a source file on any platform
        valid = tmp_path / "valid.py"
        invalid = tmp_path / "invalid.py"
        missing = tmp_path / "missing.py"
        unreadable = tmp_path / "directory.py"
        valid.write_bytes(_CONTENT.encode("utf-8"))
        invalid.write_bytes(b"value = 1\n")
        unreadable.mkdir()
        paths = [invalid, valid, unreadable, missing]

        result = subprocess.run([sys.executable, str(_CHECKER), *map(str, paths)], capture_output=True, text=True)

        output = result.stdout + result.stderr
        assert result.returncode == 1
        for path in (invalid, unreadable, missing):
            assert str(path) in output
        assert valid.read_bytes() == _CONTENT.encode("utf-8")
        assert invalid.read_bytes() == b"value = 1\n"
        assert not missing.exists()
        assert unreadable.is_dir()
