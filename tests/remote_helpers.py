"""Helpers for test on cloud files, that "mock" HTTP and S3 file storage using a local server."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
from base64 import b64encode
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
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
        if "list-type" in request.query:
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
        return web.FileResponse(directory / request.path.lstrip("/"))

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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, basic_auth: tuple[str, str] | None = None
) -> Iterator[RemoteFiles]:
    """
    Start the local HTTP/S3 server used by tests and documentation.

    Write files into the returned directory and read them through its HTTP address or s3://bucket.
    When basic_auth is supplied, files in the private subdirectory require those credentials over HTTP.
    The server stops when this context exits; monkeypatch restores reader settings when its context exits.

    :param tmp_path: Existing temporary directory in which to create the bucket and request log.
    :param monkeypatch: Pytest settings manager belonging to the test or an explicit MonkeyPatch.context().
    :param basic_auth: Optional username and password for the private subdirectory.
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
    process = context.Process(
        target=_serve_files, args=(tmp_path, request_log, child_connection, basic_auth), daemon=True
    )
    process.start()
    child_connection.close()
    try:
        if not connection.poll(20):
            raise RuntimeError("The remote file test server did not start.")
        port = connection.recv()
        files = RemoteFiles(directory=directory, endpoint=f"http://127.0.0.1:{port}", request_log=request_log)

        # Read without credentials, limit wait times, and restore settings after each test
        options = {
            "AWS_S3_ENDPOINT": f"127.0.0.1:{port}",
            "AWS_HTTPS": "NO",
            "AWS_VIRTUAL_HOSTING": "FALSE",
            "AWS_NO_SIGN_REQUEST": "YES",
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
        # Stop the mock server after the test, including when setup or reading failed
        if process.is_alive():
            connection.send("stop")
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        connection.close()


@pytest.fixture
def remote_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[RemoteFiles]:
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
    with mock_remote_files(tmp_path, monkeypatch) as files:
        yield files


@pytest.fixture(params=["http", "s3"])
def remote_url(request: pytest.FixtureRequest, remote_files: RemoteFiles) -> Callable[[str], str]:
    """Return a function that builds an HTTP or S3 address for a local test file."""
    prefix = "s3://bucket" if request.param == "s3" else f"{remote_files.endpoint}/bucket"
    query = "?token=local-test&expires=1234567890" if request.param == "http-query" else ""
    if query:
        (remote_files.directory.parent / "required-token").write_text("local-test")
    return lambda filename: f"{prefix}/{filename}{query}"
