"""Test local and remote I/O helpers across raster, vector and point cloud."""

from __future__ import annotations

import pickle
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zipfile import ZipFile

import pytest

from geoutils._io import (
    _filesystem_file_options,
    _FilesystemOpener,
    _is_remote_path,
    _path_suffix,
    _resolve_filesystem,
)
from tests.remote_helpers import RemoteFiles, mock_remote_files


class TestIsRemotePath:
    """Test module for _is_remote_path()."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("data.tif", False),
            (Path("data.tif"), False),
            (r"C:\data\image.tif", False),
            ("https://example.org/image.tif", True),
            ("s3://bucket/image.tif", True),
            ("zip://image.tif::data.zip", True),
        ],
    )
    def test_is_remote_path(self, filename: str | Path, expected: bool) -> None:
        """Checks that URLs are distinguished from local paths."""
        assert _is_remote_path(filename) is expected


class TestPathSuffix:
    """Test module for _path_suffix()."""

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("data.TIF", ".tif"),
            (Path("data.laz"), ".laz"),
            ("https://example.org/data.TIF?token=a.b#name.txt", ".tif"),
            ("s3://bucket/data.parquet", ".parquet"),
            ("https://example.org/points/?token=a.b", ""),
        ],
    )
    def test_path_suffix(self, filename: str | Path, expected: str) -> None:
        """Checks that file extensions are lowercase and exclude URL properly."""
        assert _path_suffix(filename) == expected


class TestResolveFilesystem:
    """Test module for _resolve_filesystem()."""

    def test_resolve_filesystem__local_paths(self, tmp_path: Path) -> None:
        """Checks that local paths become strings (and don't create a filesystem)."""
        path = tmp_path / "data.bin"

        class LocalPath:
            """Local filename through the path protocol."""

            def __fspath__(self) -> str:
                return str(path)

        for source in (str(path), path, LocalPath()):
            assert _resolve_filesystem(source) == (str(path), None)

    def test_resolve_filesystem__configured_path(self, tmp_path: Path) -> None:
        """Checks that a configured path provides both its path and filesystem."""

        fsspec = pytest.importorskip("fsspec")
        path = tmp_path / "data.bin"
        path.write_bytes(b"point values")

        class ConfiguredPath:
            """Relative path with configured filesystem."""
            def __init__(self) -> None:
                self.path = path.name
                self.fs = fsspec.filesystem("dir", path=str(tmp_path), fs=fsspec.filesystem("file"))

        source = ConfiguredPath()
        filename, filesystem = _resolve_filesystem(source)

        assert filename == path.name
        assert filesystem is source.fs
        assert filesystem.cat(filename) == path.read_bytes()

    @pytest.mark.parametrize("remote_default", [False, True])
    @pytest.mark.parametrize("storage_options", [None, {}])
    def test_resolve_filesystem__url_dispatch(
        self, remote_default: bool, storage_options: dict[str, Any] | None, tmp_path: Path
    ) -> None:
        """Checks that URL dispatch follows remote_default unless access options are passed."""
        fsspec = pytest.importorskip("fsspec")
        path = str(tmp_path / "data.bin")
        source = "memory://" + path
        fsspec.filesystem("memory").pipe(path, b"point values")

        filename, filesystem = _resolve_filesystem(
            source, storage_options=storage_options, remote_default=remote_default
        )

        if remote_default or storage_options is not None:
            assert filename == path
            assert filesystem.cat(filename) == b"point values"
        else:
            assert filename == source
            assert filesystem is None

    @pytest.mark.parametrize("filename", ["data.bin", "memory://data.bin"])
    def test_resolve_filesystem__explicit_filesystem(self, filename: str) -> None:
        """Checks that an explicit filesystem is returned with the path."""
        filesystem = pytest.importorskip("fsspec").filesystem("memory")
        result_path, result_filesystem = _resolve_filesystem(filename, filesystem=filesystem)

        assert result_path == filename
        assert result_filesystem is filesystem

    @pytest.mark.parametrize("protocol", ["http", "s3"])
    @pytest.mark.parametrize("access", ["filesystem", "storage_options"])
    def test_resolve_filesystem__access_options(
        self, protocol: str, access: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Checks that HTTP and S3 access options open the requested file."""

        fsspec = pytest.importorskip("fsspec")
        with mock_remote_files(tmp_path, monkeypatch, basic_auth=("demo", "geoutils")) as files:
            # Write test data
            directory = files.directory / "private" if protocol == "http" else files.directory
            directory.mkdir(exist_ok=True)
            (directory / "data.bin").write_bytes(b"point values")

            # Set access options
            options: dict[str, Any]
            if protocol == "http":
                source = f"{files.endpoint}/bucket/private/data.bin"
                options = {"headers": {"Authorization": "Basic ZGVtbzpnZW91dGlscw=="}}
            else:
                monkeypatch.delitem(fsspec.config.conf, "s3", raising=False)
                monkeypatch.delenv("FSSPEC_S3", raising=False)
                source = "s3://bucket/data.bin"
                options = {"anon": True, "endpoint_url": files.endpoint}
            access_options = (
                {"filesystem": fsspec.filesystem(protocol, **options)}
                if access == "filesystem"
                else {"storage_options": options}
            )

            filename, filesystem = _resolve_filesystem(source, **access_options)

            with filesystem.open(filename, "rb") as stream:
                assert stream.read() == b"point values"

    def test_resolve_filesystem__file_in_zip(self, tmp_path: Path) -> None:
        """Checks that a ZIP URL opens the named file inside the ZIP archive."""
        pytest.importorskip("fsspec")

        # Create and write a file
        path = tmp_path / "values.txt"
        path.write_text("point values")

        # Add it to a ZIP archive
        zip_path = tmp_path / "data.zip"
        with ZipFile(zip_path, "w") as archive:
            archive.write(path, path.name)

        # Open the file inside the ZIP
        filename, filesystem = _resolve_filesystem(f"zip://{path.name}::{zip_path}", storage_options={})

        assert filename == path.name
        with filesystem.open(filename) as stream:
            assert stream.read() == path.read_bytes()
        filesystem.close()

    def test_resolve_filesystem__existing_object(self) -> None:
        """Checks that an existing object is returned unchanged without a filesystem."""
        source = object()
        result, filesystem = _resolve_filesystem(source)

        assert result is source
        assert filesystem is None


class TestFilesystemOpener:
    """Test module for _FilesystemOpener.open()."""

    @pytest.mark.parametrize("mode", ["r", "rt", "rb"])
    def test_open__binary_read(self, mode: str, tmp_path: Path) -> None:
        """Checks that opening a file returns bytes for text and binary modes."""
        filesystem = pytest.importorskip("fsspec").filesystem("file")
        path = tmp_path / "data.bin"
        path.write_bytes(b"point values")
        opener = _FilesystemOpener(filesystem)

        with opener.open(str(path), mode) as stream:
            assert stream.read() == path.read_bytes()
        assert stream.stream.closed
        assert opener.size(str(path)) == path.stat().st_size

    def test_open__independent_handles(self, tmp_path: Path) -> None:
        """Checks that repeated opens have distinct identities and independent file positions."""
        filesystem = pytest.importorskip("fsspec").filesystem("file")
        path = tmp_path / "data.bin"
        path.write_bytes(b"point values")
        opener = _FilesystemOpener(filesystem)

        with opener.open(str(path)) as first, opener.open(str(path)) as second:
            assert len({first, second}) == 2
            assert first.read(1) == second.read(1) == b"p"
            first.seek(5)
            assert second.tell() == 1

    def test_open__binary_write(self, tmp_path: Path) -> None:
        """Checks that text write mode accepts a buffer and closes the file."""
        filesystem = pytest.importorskip("fsspec").filesystem("file")
        path = tmp_path / "data.bin"
        opener = _FilesystemOpener(filesystem)
        values = b"point values"

        with opener.open(str(path), "wt") as stream:
            assert stream.write(memoryview(values)) == len(values)

        assert stream.stream.closed
        assert path.read_bytes() == values


class TestFilesystemFileOptions:
    """Test module for _filesystem_file_options()."""

    def test_filesystem_file_options__local_path(self, tmp_path: Path) -> None:
        """Checks that a local path uses the user filesystem without changing the filename."""
        filesystem = pytest.importorskip("fsspec").filesystem("file")
        path = tmp_path / "data.bin"
        filename, opener = _filesystem_file_options(path, filesystem)

        assert filename == str(path)
        assert opener.filesystem is filesystem

    @pytest.mark.parametrize("remote_url", ["http-query"], indirect=True)
    def test_filesystem_file_options__companion_files(
        self, remote_files: RemoteFiles, remote_url: Callable[[str], str]
    ) -> None:
        """Checks that a file and its companion file use the same URL query."""
        filesystem = pytest.importorskip("fsspec").filesystem("http")
        (remote_files.directory / "image.tif").write_bytes(b"pixels")
        (remote_files.directory / "image.tif.msk").write_bytes(b"mask")

        filename, opener = _filesystem_file_options(remote_url("image.tif"), filesystem)

        assert filename == "image.tif"
        for name, expected in ((filename, b"pixels"), (filename + ".msk", b"mask")):
            with opener.open(name) as stream:
                assert stream.read() == expected


class TestRestoreURLDirectory:
    """Test module for _restore_url_directory()."""

    @pytest.mark.parametrize("remote_url", ["http-query"], indirect=True)
    def test_restore_url_directory__pickle(self, remote_files: RemoteFiles, remote_url: Callable[[str], str]) -> None:
        """Checks that serialization preserves the directory and URL query for companion files."""
        filesystem = pytest.importorskip("fsspec").filesystem("http")
        (remote_files.directory / "image.tif.msk").write_bytes(b"mask")
        filename, opener = _filesystem_file_options(remote_url("image.tif"), filesystem)

        directory = pickle.loads(pickle.dumps(opener.filesystem))

        with directory.open(filename + ".msk", "rb") as stream:
            assert stream.read() == b"mask"


class TestResolveFilesystemErrors:
    """Test module for warnings/error in _resolve_filesystem()."""

    @pytest.mark.parametrize(
        "source,options,error,match",
        [
            (
                "data.bin",
                {"filesystem": object(), "storage_options": {}},
                ValueError,
                "either filesystem or storage_options",
            ),
            ("data.bin", {"filesystem": object()}, TypeError, "fsspec-compatible filesystem"),
            (object(), {"filesystem": object()}, ValueError, "require a path or URL"),
            (object(), {"storage_options": {}}, ValueError, "require a path or URL"),
        ],
    )
    def test_resolve_filesystem__error_access_options(
        self, source: Any, options: dict[str, Any], error: type[Exception], match: str
    ) -> None:
        """Checks an error is raised for conflicting access options, an invalid filesystem or a missing path."""
        with pytest.raises(error, match=match):
            _resolve_filesystem(source, **options)

    @pytest.mark.parametrize("access", ["filesystem", "storage_options"])
    def test_resolve_filesystem__error_configured_path(self, access: str) -> None:
        """Checks an error is raised when a configured path also receives access options."""
        filesystem = pytest.importorskip("fsspec").filesystem("memory")
        source = SimpleNamespace(path="data.bin", fs=filesystem)
        options: dict[str, Any] = {access: filesystem if access == "filesystem" else {}}

        with pytest.raises(ValueError, match="settings on the path or as arguments"):
            _resolve_filesystem(source, **options)
