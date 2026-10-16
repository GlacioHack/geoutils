"""Test the local server shared by remote file tests and documentation."""

from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPBasicAuthHandler, HTTPPasswordMgrWithDefaultRealm, Request, build_opener, urlopen

import pytest

from tests.remote_helpers import mock_remote_files


@pytest.mark.parametrize(
    "credentials", [None, ("demo", "incorrect"), ("demo", "geoutils")], ids=["missing", "incorrect", "matching"]
)
def test_mock_remote_files__password_protected_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, credentials: tuple[str, str] | None
) -> None:
    """Checks that private files require matching credentials and return the requested byte range."""

    # Create public and private copies of ten bytes so the selected range has a known value
    with mock_remote_files(tmp_path, monkeypatch, basic_auth=("demo", "geoutils")) as files:
        contents = b"0123456789"
        (files.directory / "public.bin").write_bytes(contents)
        private = files.directory / "private"
        private.mkdir()
        (private / "values.bin").write_bytes(contents)
        private_url = f"{files.endpoint}/bucket/private/values.bin"

        # Read the public file without credentials and configure a standard HTTP Basic client
        with urlopen(f"{files.endpoint}/bucket/public.bin", timeout=5) as response:
            assert response.read() == contents
        passwords = HTTPPasswordMgrWithDefaultRealm()
        if credentials is not None:
            passwords.add_password(None, private_url, *credentials)
        client = build_opener(HTTPBasicAuthHandler(passwords))
        request = Request(private_url, headers={"Range": "bytes=2-4"})

        # Check that private reads need the correct password and preserve the requested range
        if credentials != ("demo", "geoutils"):
            with pytest.raises(HTTPError) as error:
                client.open(request, timeout=5)
            assert error.value.code == 401
        else:
            with client.open(request, timeout=5) as response:
                assert response.status == 206
                assert response.headers["Content-Range"] == "bytes 2-4/10"
                assert response.read() == b"234"
