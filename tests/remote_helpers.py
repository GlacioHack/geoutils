"""Helpers for test on cloud files, that "mock" HTTP and S3 file storage using a local server."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
from base64 import b64encode
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from functools import wraps
from multiprocessing.connection import Connection
from multiprocessing.pool import Pool
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pytest


@dataclass
class RemoteFiles:
    """This class defines local test files and request history to mock HTTP and S3 storage."""

    directory: Path
    endpoint: str
    request_log: Path

    @property
    def requests(self) -> list[tuple[str, str, str | None]]:
        """List requests received by the mock server, including requests for missing files."""

        return [tuple(json.loads(line)) for line in self.request_log.read_text().splitlines()]


@contextmanager
def _remote_filesystem_sessions(fsspec: Any) -> Iterator[None]:
    """Close HTTP and S3 sessions created during a remote file test."""

    from fsspec.asyn import sync

    # Track construction so uncached filesystems are closed properly
    instances = []
    with pytest.MonkeyPatch.context() as patch:
        for protocol in ("http", "s3"):
            filesystem_class = fsspec.get_filesystem_class(protocol)

            @wraps(filesystem_class.__init__)
            def record_filesystem(
                filesystem: Any,
                *args: Any,
                _initialize: Callable[..., None] = filesystem_class.__init__,
                **kwargs: Any,
            ) -> None:
                """Record a newly initialized filesystem for session cleanup."""

                _initialize(filesystem, *args, **kwargs)
                instances.append(filesystem)

            patch.setattr(filesystem_class, "__init__", record_filesystem)

        try:
            yield
        finally:
            # Await every close while the server is still available, even if another close fails
            with ExitStack() as cleanup:
                for filesystem in instances:
                    s3_context = getattr(filesystem, "_s3creator", None)
                    http_session = getattr(filesystem, "_session", None)
                    if s3_context is not None:
                        cleanup.callback(sync, filesystem.loop, s3_context.__aexit__, None, None, None, timeout=10)
                    elif http_session is not None:
                        cleanup.callback(sync, filesystem.loop, http_session.close, timeout=10)

                    # Remove only owned instances from the cache so later tests cannot reuse closed sessions
                    cache = type(filesystem)._cache
                    for token, cached in tuple(cache.items()):
                        if cached is filesystem:
                            del cache[token]


def _serve_files(
    directory: Path,
    request_log: Path,
    connection: Connection,
    basic_auth: tuple[str, str] | None = None,
) -> None:
    """
    Run the mock server in a separate process so it can answer requests while GDAL reads files.

    respond() records requests and lists files belonging to a dataset.
    FileResponse() returns file information
    and all or part of a file: https://docs.aiohttp.org/en/stable/web_reference.html#aiohttp.web.FileResponse
    run_server() starts the mock server and waits for the test fixture to stop it.
    """
    from aiohttp import web

    async def respond(request: web.Request) -> web.StreamResponse:
        """Record a request and return a file list, file information, or file contents."""

        # Log all requests, including GDAL's checks for extra files beside the dataset
        with request_log.open("a") as log:
            log.write(json.dumps((request.method, request.raw_path, request.headers.get("Range"))) + "\n")

        # Check the URL's access token to catch readers that drop it
        required_token = directory / "required-token"
        if required_token.is_file() and request.query.get("token") != required_token.read_text():
            raise web.HTTPForbidden()

        # Require matching credentials for private files when authentication is enabled
        if basic_auth is not None and request.path.startswith("/bucket/private/"):
            authorization = "Basic " + b64encode(":".join(basic_auth).encode()).decode("ascii")
            if request.headers.get("Authorization") != authorization:
                raise web.HTTPUnauthorized(headers={"WWW-Authenticate": 'Basic realm="GeoUtils example"'})

        # List matching files so readers can find datasets split across several files
        # Response fields follow https://docs.aws.amazon.com/AmazonS3/latest/API/API_ListObjectsV2.html
        # GDAL 3.12 in Python 3.14 CI uses ListObjects to check GeoPackage companion files
        # Its requests have a prefix but no list-type parameter
        if "list-type" in request.query or "prefix" in request.query:
            prefix = request.query.get("prefix", "")
            bucket = directory / "bucket"
            listing = ElementTree.Element("ListBucketResult", xmlns="http://s3.amazonaws.com/doc/2006-03-01/")
            ElementTree.SubElement(listing, "Name").text = "bucket"
            ElementTree.SubElement(listing, "Prefix").text = prefix
            ElementTree.SubElement(listing, "Delimiter").text = request.query.get("delimiter", "")
            ElementTree.SubElement(listing, "IsTruncated").text = "false"

            # Use a fixed order and date so repeated file listings match
            count = 0
            for candidate in sorted(bucket.rglob("*")):
                key = candidate.relative_to(bucket).as_posix()
                if not candidate.is_file() or not key.startswith(prefix):
                    continue
                entry = ElementTree.SubElement(listing, "Contents")
                ElementTree.SubElement(entry, "Key").text = key
                ElementTree.SubElement(entry, "Size").text = str(candidate.stat().st_size)
                ElementTree.SubElement(entry, "LastModified").text = "2026-01-01T00:00:00.000Z"
                count += 1

            # Return the file list (aiohttp handles requests to read individual files)
            ElementTree.SubElement(listing, "KeyCount").text = str(count)
            return web.Response(body=ElementTree.tostring(listing), content_type="application/xml")

        # List files and subdirectories for HTTP readers
        path = directory / request.path.lstrip("/")
        if path.is_dir():
            listing = ElementTree.Element("html")
            for candidate in sorted(path.iterdir()):
                name = candidate.name + ("/" if candidate.is_dir() else "")
                ElementTree.SubElement(listing, "a", href=name).text = name
            return web.Response(body=ElementTree.tostring(listing, method="html"), content_type="text/html")
        return web.FileResponse(path)

    async def run_server() -> None:
        """Start the mock server and wait for the test fixture to stop it."""

        # Start a local server that passes every request to respond()
        app = web.Application()
        app.router.add_route("*", "/{path:.*}", respond)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        try:
            # Wait for the stop message without blocking file requests
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            connection.send(runner.addresses[0][1])
            await asyncio.to_thread(connection.recv)
        finally:
            await runner.cleanup()

    # Close the connection to the test fixture, even if the server failed to start
    try:
        asyncio.run(run_server())
    finally:
        connection.close()


@contextmanager
def mock_remote_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    basic_auth: tuple[str, str] | None = None,
    pool: Pool | None = None,
) -> Iterator[RemoteFiles]:
    """
    Start the local HTTP/S3 server used by tests and documentation.

    Write files into the returned directory and read them through its HTTP address or s3://bucket.
    When basic_auth is supplied, files in the private subdirectory require those credentials over HTTP.
    The server stops when this context exits; monkeypatch restores reader settings when its context exits.

    :param tmp_path: Existing temporary directory in which to create the bucket and request log.
    :param monkeypatch: Pytest settings manager belonging to the test or an explicit MonkeyPatch.context().
    :param basic_auth: Optional username and password for the private subdirectory.
    :param pool: Optional worker pool to reuse the server process between tests.
    :returns: The directory, server address and request history available inside the context.
    """

    # Skip these tests if an optional library needed for remote reads is missing
    pytest.importorskip("aiohttp")
    fsspec = pytest.importorskip("fsspec")
    pytest.importorskip("s3fs")

    # Create the local folder used for both HTTP and S3 reads
    directory = tmp_path / "bucket"
    directory.mkdir()
    request_log = tmp_path / "requests.jsonl"
    request_log.touch()

    # Give each test its own server address to avoid interference between tests
    context = multiprocessing.get_context("spawn")
    connection, child_connection = context.Pipe()
    pool_context = nullcontext(pool) if pool is not None else context.Pool(processes=1)
    with pool_context as server_pool:
        server = server_pool.apply_async(_serve_files, (tmp_path, request_log, child_connection, basic_auth))
        try:
            if not connection.poll(20):
                raise RuntimeError("The remote file test server did not start.")
            port = connection.recv()
            child_connection.close()
            files = RemoteFiles(directory=directory, endpoint=f"http://127.0.0.1:{port}", request_log=request_log)

            with _remote_filesystem_sessions(fsspec):
                # Read without credentials, limit wait times, and restore settings after each test
                options = {
                    "AWS_S3_ENDPOINT": f"127.0.0.1:{port}",
                    "AWS_HTTPS": "NO",
                    "AWS_VIRTUAL_HOSTING": "FALSE",
                    "AWS_NO_SIGN_REQUEST": "YES",
                    # Discard cached S3 contents on close because tests reuse the same bucket paths
                    "CPL_VSIL_CURL_NON_CACHED": "/vsis3/bucket/",
                    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
                    "GDAL_HTTP_TIMEOUT": "5",
                    "GDAL_HTTP_MAX_RETRY": "0",
                    "NO_PROXY": "127.0.0.1,localhost",
                    "no_proxy": "127.0.0.1,localhost",
                }
                for name, value in options.items():
                    monkeypatch.setenv(name, value)

                # Configure point cloud readers to use the same local storage through fsspec
                s3_options = {"anon": True, "endpoint_url": files.endpoint}
                monkeypatch.setitem(fsspec.config.conf, "s3", s3_options)
                monkeypatch.setenv("FSSPEC_S3", json.dumps(s3_options))
                yield files
        finally:
            # Stop this listener before another test uses the same process
            try:
                if not server.ready():
                    connection.send("stop")
                server.get(timeout=5)
            except multiprocessing.TimeoutError:
                server_pool.terminate()
                raise RuntimeError("The remote file test server did not stop.") from None
            finally:
                child_connection.close()
                connection.close()


@pytest.fixture(scope="module")
def remote_pool() -> Iterator[Pool]:
    """Reuse a spawned server process without sharing files or listeners between tests."""

    pytest.importorskip("aiohttp")
    with multiprocessing.get_context("spawn").Pool(processes=1) as pool:
        yield pool


@pytest.fixture
def remote_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remote_pool: Pool) -> Iterator[RemoteFiles]:
    """
    Mock remote storage for tests using temporary local files.

    To use this fixture and remote_url() in a test:
    - Write inputs under remote_files.directory,
    - Open them with remote_url("example.tif"),
    - Inspect remote_files.requests to check which reads were made.

    The fixture configures readers to use the mock server and stops it after the test.

    A real local server tests file reads made by GDAL as well as Python libraries.
    This follows GDAL's S3 tests: https://github.com/OSGeo/gdal/blob/master/autotest/gcore/vsis3.py
    """
    with mock_remote_files(tmp_path, monkeypatch, pool=remote_pool) as files:
        yield files


@pytest.fixture(params=["http", "s3"])
def remote_url(request: pytest.FixtureRequest, remote_files: RemoteFiles) -> Callable[[str], str]:
    """Return a function that builds an HTTP or S3 address for a local test file."""
    prefix = "s3://bucket" if request.param == "s3" else f"{remote_files.endpoint}/bucket"
    query = "?token=local-test&expires=1234567890" if request.param == "http-query" else ""
    if query:
        (remote_files.directory.parent / "required-token").write_text("local-test")
    return lambda filename: f"{prefix}/{filename}{query}"


@pytest.fixture
def file_access(
    request: pytest.FixtureRequest, filename: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remote_pool: Pool
) -> Iterator[tuple[Path, str, dict[str, Any]]]:
    """Write test data and provide HTTP or S3 access options."""

    import numpy as np
    import rasterio as rio
    from rasterio.transform import from_origin

    import geoutils as gu

    fsspec = pytest.importorskip("fsspec")
    if filename.endswith(".gpkg"):
        pytest.importorskip("fiona")
    with mock_remote_files(tmp_path, monkeypatch, basic_auth=("demo", "geoutils"), pool=remote_pool) as files:
        # Write test data
        path = files.directory / "private" / filename
        path.parent.mkdir()
        if filename.endswith(".tif"):
            values = np.ma.array(np.arange(96, dtype=np.int16).reshape(8, 12), mask=False)
            values.mask[0, 0] = True
            gu.Raster.from_array(values, from_origin(0, 8, 1, 1), 32633, nodata=-9999).to_file(path)
            with rio.open(path, "r+") as raster:
                raster.build_overviews([2, 4], rio.enums.Resampling.average)
        else:
            points = gu.PointCloud.from_xyz(
                np.arange(7.0), np.arange(7.0) + 1, np.arange(7.0) + 10, 32633, data_name="height"
            )
            if filename.endswith((".las", ".laz")):
                points.to_las(path)
            else:
                points.gdf["quality"] = np.arange(7)
                points.gdf["label"] = ["ground", "tree"] * 3 + ["ground"]
                if filename.endswith(".gpkg"):
                    points.to_file(path)
                else:
                    points.to_parquet(path, chunks=3, partitioned=filename == "parts")

        # Set access options
        protocol, option = request.param.split("-")
        options: dict[str, Any]
        if protocol == "http":
            source = f"{files.endpoint}/bucket/private/{path.name}"
            options = {"headers": {"Authorization": "Basic ZGVtbzpnZW91dGlscw=="}}
        else:
            (files.directory / path.name).write_bytes(path.read_bytes())
            monkeypatch.delitem(fsspec.config.conf, "s3", raising=False)
            monkeypatch.delenv("FSSPEC_S3", raising=False)
            source = f"bucket/{path.name}"
            options = {"anon": True, "endpoint_url": files.endpoint}
            if option == "storage_options":
                source = "s3://" + source
        value = fsspec.filesystem(protocol, **options) if option == "filesystem" else options
        yield path, source, {option: value}
