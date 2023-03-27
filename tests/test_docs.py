"""Functions to test the documentation."""
import os
import platform
import shutil

import sphinx.cmd.build


class TestDocs:
    docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../", "doc/")
    n_threads = os.getenv("N_CPUS")

    def test_build(self) -> None:
        """Try building the docs and see if it works."""
        # Remove the build directory if it exists.

        # Test only on Linux
        if platform.system() == "Linux":
            # Remove the build directory if it exists.
            if os.path.isdir(os.path.join(self.docs_dir, "build/")):
                shutil.rmtree(os.path.join(self.docs_dir, "build/"))

            return_code = sphinx.cmd.build.main(
                [
                    "-j",
                    "1",
                    os.path.join(self.docs_dir, "source/"),
                    os.path.join(self.docs_dir, "build/html"),
                ]
            )

            assert return_code == 0
