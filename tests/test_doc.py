"""Functions to test the documentation."""
import os
import platform
import shutil
import warnings

import sphinx.cmd.build


class TestDocs:
    docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../", "doc/")
    n_threads = os.getenv("N_CPUS")

    def test_build(self) -> None:
        """Try building the documentation and see if it works."""

        # Ignore all user warnings raised in the documentation
        # (some are shown on purpose in certain examples, so they shouldn't make the test fail)
        warnings.filterwarnings("ignore", category=UserWarning)

        # Building the doc fails on Windows for the CLI section
        if (platform.system() == "Linux") or (platform.system() == "Darwin"):

            # Remove the build directory if it exists.
            if os.path.isdir(os.path.join(self.docs_dir, "build/")):
                shutil.rmtree(os.path.join(self.docs_dir, "build/"))

            return_code = sphinx.cmd.build.main(
                [
                    "-j",
                    "1",
                    os.path.join(self.docs_dir, "source/"),
                    os.path.join(self.docs_dir, "build/"),
                ]
            )

            assert return_code == 0
