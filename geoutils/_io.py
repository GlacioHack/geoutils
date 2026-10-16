"""Shared functions for local/remote I/O."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from geoutils._misc import import_optional


def _is_remote_path(filename: str | Path) -> bool:
    """Identify URLs while leaving local paths, including Windows drive letters, unchanged."""
    return "://" in os.fspath(filename)


def _path_suffix(filename: str | Path) -> str:
    """Read a file extension without including a remote URL's query string or fragment."""
    path = urlsplit(os.fspath(filename)).path if _is_remote_path(filename) else os.fspath(filename)
    return Path(path).suffix.lower()


def _get_filesystem(filename: str | Path) -> Any:
    """
    Resolve a remote filesystem with fsspec's credentials and configuration, or use local file access.

    URL dispatch follows https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.core.url_to_fs.
    Credentials and backend options come from https://filesystem-spec.readthedocs.io/en/latest/config.html.
    """
    if not _is_remote_path(filename):
        return None
    fsspec = import_optional("fsspec")
    return fsspec.core.url_to_fs(os.fspath(filename))[0]
