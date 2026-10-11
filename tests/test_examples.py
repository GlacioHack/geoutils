"""
Test the example files used for testing and documentation
"""

import hashlib
import io
import tarfile
import warnings
from pathlib import Path
from unittest.mock import Mock

import pytest

import geoutils as gu
from geoutils import examples


@pytest.mark.parametrize(
    "example", ["everest_landsat_b4", "everest_landsat_b4_cropped", "everest_landsat_rgb", "exploradores_aster_dem"]
)
def test_read_paths_raster(example: str) -> None:
    assert isinstance(gu.Raster(examples.get_path(example)), gu.Raster)
    assert isinstance(gu.Raster(examples.get_path_test(example)), gu.Raster)


@pytest.mark.parametrize("example", ["everest_rgi_outlines", "exploradores_rgi_outlines"])
def test_read_paths_vector(example: str) -> None:
    warnings.simplefilter("error")
    assert isinstance(gu.Vector(examples.get_path(example)), gu.Vector)
    assert isinstance(gu.Vector(examples.get_path_test(example)), gu.Vector)


class TestDownloadExamples:
    """Test module for downloading the example archive and reporting download failures."""

    def test_download_examples(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks downloading (without network access, using some monkeypatching)."""

        # We build an archive with the same example directories
        directory_names = ("Everest_Landsat", "Exploradores_ASTER", "Coromandel_Lidar")
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:gz") as tar:
            for name in directory_names:
                content = name.encode()
                member = tarfile.TarInfo(f"geoutils-data-test/data/{name}/example.txt")
                member.size = len(content)
                tar.addfile(member, io.BytesIO(content))

        # The, we mock the download and installation
        response = Mock()
        response.getcode.return_value = 200
        response.read.return_value = archive.getvalue()
        urlopen = Mock(return_value=response)
        destination = tmp_path / "examples"
        destination.mkdir()
        monkeypatch.setattr(examples.urllib.request, "urlopen", urlopen)
        monkeypatch.setattr(examples, "_EXAMPLES_DIRECTORY", destination)

        # Check the download of every directory
        examples.download_examples(overwrite=True)
        for name in directory_names:
            assert (destination / name / "example.txt").read_bytes() == name.encode()
        urlopen.assert_called_once()

    def test_download_examples__error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Checks an error is raised when the download of the examples fail."""

        # We mock a failed HTTP response
        response = Mock(status_code=503)
        response.getcode.return_value = 503
        monkeypatch.setattr(examples.urllib.request, "urlopen", Mock(return_value=response))
        destination = tmp_path / "examples"
        destination.mkdir()
        monkeypatch.setattr(examples, "_EXAMPLES_DIRECTORY", destination)

        # We check the error is properly raised
        with pytest.raises(ValueError, match="non-200 response: 503"):
            examples.download_examples(overwrite=True)
        assert list(destination.iterdir()) == []


# Original sha256 obtained with `sha256sum filename`
original_sha256_examples = {
    "everest_landsat_b4": "271fa34e248f016f87109c8e81960caaa737558fbae110ec9e0d9e2d30d80c26",
    "everest_landsat_b4_cropped": "0e63d8e9c4770534a1ec267c91e80cd9266732184a114f0bd1aadb5a613215e6",
    "everest_landsat_rgb": "7d0505a8610fd7784cb71c03e5b242715cd1574e978c2c86553d60fd82372c30",
    "everest_rgi_outlines": "d1a5bcd4bd4731a24c2398c016a6f5a8064160fedd5bab10609adacda9ba41ef",
    "exploradores_aster_dem": "dcb0d708d042553cdd2bb4fd82c55b5674a5e0bd6ea46f1a021b396b7d300033",
    "exploradores_rgi_outlines": "19c2dac089ce57373355213fdf2fd72f601bf97f21b04c4920edb1e4384ae2b2",
    "coromandel_lidar": "2f1fff1bb84860a8438e14d39e14bf974236dc6345e64649a131507d0ed844f3",
}


@pytest.mark.parametrize("example", examples.available)
def test_data_integrity__examples(example: str) -> None:
    """
    Test that input data is not corrupted by checking sha265 sum
    """
    # Read file as bytes
    fbytes = open(examples.get_path(example), "rb").read()

    # Get sha256
    file_sha256 = hashlib.sha256(fbytes).hexdigest()

    assert file_sha256 == original_sha256_examples[example]


original_sha256_test = {
    "everest_landsat_b4": "5aa1a0a1c17efd211e42218ab5e2f3e0e404b96ba5055ac5eebef756ad5c65bc",
    "everest_landsat_b4_cropped": "4244767c31c51f7c7b5fb8eb48df7d6394aa707deb9fe699d5672ff9d2507aef",
    "everest_landsat_rgb": "b77109f8027418cdd36ccab34cc3996bbbf2756b116ecf0fed8e4163cd7aa2f9",
    "everest_rgi_outlines": "3642e2fa5da1d9cad2378e0941985ae47077dba7839e30fdd413f8e868ab2ade",
    "exploradores_aster_dem": "c98f24cb131810dd8b2f4773a8df0821cf31edec79890967a67b8a6fdb89314d",
    "exploradores_rgi_outlines": "2f0281b00a49ad2f0874fb4ee54df1e0d11ad073f826d9ca713430588c15fa15",
    "coromandel_lidar": "95af5de14205c712e7674723d00119f4fa6239a65fb2aa3f7035254ace3194ae",
}


@pytest.mark.parametrize("example_test", examples.available_test)
def test_data_integrity__tests(example_test: str) -> None:
    """
    Test that input data is not corrupted by checking sha265 sum
    """
    # Read file as bytes
    fbytes = open(examples.get_path_test(example_test), "rb").read()

    # Get sha256
    file_sha256 = hashlib.sha256(fbytes).hexdigest()

    assert file_sha256 == original_sha256_test[example_test]
