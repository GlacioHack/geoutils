"""Shared functions for local/remote I/O for all objects."""

from __future__ import annotations

import os
import posixpath
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from geoutils._misc import import_optional


class _FilesystemFile:
    """Small class to give each GDAL file handle its own identity when forwarding file operations."""

    def __init__(self, stream: Any) -> None:
        self.stream = stream

    def __getattr__(self, name: str) -> Any:
        return getattr(self.stream, name)

    def write(self, data: Any) -> int:
        return self.stream.write(bytes(data))

    def __enter__(self) -> _FilesystemFile:
        self.stream.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        return self.stream.__exit__(*args)


class _FilesystemOpener:
    """
    A small helper class for opening the same file several times with GDAL.

    Some filesystems consider two open files equal when their paths match. We wrap each open file so GDAL can
    track and close it separately.
    """

    def __init__(self, filesystem: Any) -> None:
        self.filesystem = filesystem

    def __getattr__(self, name: str) -> Any:
        filesystem = self.__dict__.get("filesystem")
        if filesystem is None:
            raise AttributeError(name)
        return getattr(filesystem, name)

    def open(self, path: str, mode: str = "rb", **kwargs: Any) -> Any:
        mode = mode.replace("t", "")
        if "b" not in mode:
            mode += "b"
        return _FilesystemFile(self.filesystem.open(path, mode, **kwargs))


def _filesystem_file_options(filename: str | Path, filesystem: Any) -> tuple[str, _FilesystemOpener]:
    """
    Give GDAL a filename whose companion files use the same URL query.

    Shared by Raster, Vector, PointCloud and their accessors.
    """
    filename = os.fspath(filename)
    if not _is_remote_path(filename):
        return filename, _FilesystemOpener(filesystem)

    import_optional("fsspec")
    from fsspec.implementations.dirfs import DirFileSystem

    # GDAL appends companion suffixes to filenames, before any URL query
    url = urlsplit(filename)
    directory = url._replace(path=posixpath.dirname(url.path), query="", fragment="").geturl()

    class URLDirectory(DirFileSystem):  # type: ignore[misc]
        """Attach the URL query after joining filenames to their directory."""

        def _join(self, path: Any) -> Any:
            """Join a GDAL filename and then add the access query."""
            joined = super()._join(path)
            return joined + "?" + url.query if url.query and isinstance(joined, str) else joined

        def __reduce__(self) -> tuple[Any, ...]:
            """Recreate the directory without serializing Fsspec's class cache."""
            return _restore_url_directory, (filename, filesystem)

    directory_filesystem = URLDirectory(directory, fs=filesystem, skip_instance_cache=True)
    return posixpath.basename(url.path), _FilesystemOpener(directory_filesystem)


def _restore_url_directory(filename: str, filesystem: Any) -> Any:
    """Restore URL access when a worker unpickles the directory filesystem."""
    return _filesystem_file_options(filename, filesystem)[1].filesystem


def _is_remote_path(filename: str | Path) -> bool:
    """Identify remote paths from local paths."""
    return "://" in os.fspath(filename)


def _path_suffix(filename: str | Path) -> str:
    """Read a file extension properly for a remote path."""
    path = urlsplit(os.fspath(filename)).path if _is_remote_path(filename) else os.fspath(filename)
    return Path(path).suffix.lower()


def _resolve_filesystem(
    filename: Any,
    filesystem: Any = None,
    storage_options: dict[str, Any] | None = None,
    *,
    remote_default: bool = True,
) -> tuple[Any, Any]:
    """
    Check user input for a remote filesystem with fsspec's credentials and configuration.

    URL dispatch follows https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.core.url_to_fs.
    Credentials and backend options come from https://filesystem-spec.readthedocs.io/en/latest/config.html.

    The output is the resolved path, and the filesystem.

    Shared by Raster, Vector, PointCloud and their accessors.
    """
    if filesystem is not None and storage_options is not None:
        raise ValueError("Pass either filesystem or storage_options, not both.")

    # Configured path objects can expose both their path and filesystem
    if isinstance(getattr(filename, "path", None), str) and callable(
        getattr(getattr(filename, "fs", None), "open", None)
    ):
        if filesystem is not None or storage_options is not None:
            raise ValueError("Pass filesystem settings on the path or as arguments, not both.")
        filesystem = filename.fs
        filename = filename.path
    if isinstance(filename, os.PathLike):
        filename = os.fsdecode(filename)
    if not isinstance(filename, str):
        if filesystem is not None or storage_options is not None:
            raise ValueError("Filesystem settings require a path or URL.")
        return filename, None

    # We resolve URLs once and preserve the path inside archives and wrapped filesystems
    if filesystem is not None:
        if not callable(getattr(filesystem, "open", None)):
            raise TypeError("Argument 'filesystem' must be an fsspec-compatible filesystem with open().")
        return filename, filesystem
    if storage_options is None and (not remote_default or not _is_remote_path(filename)):
        return filename, None
    fsspec = import_optional("fsspec")
    resolved_filesystem, path = fsspec.core.url_to_fs(filename, **(storage_options or {}))
    return path, resolved_filesystem
