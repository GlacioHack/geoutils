"""Functions to test the documentation."""
import os
import platform
import shutil
import warnings

import sphinx.cmd.build


class TestDocs:
    docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../", "doc/")
    n_threads = os.getenv("N_CPUS")

    def test_example_code(self) -> None:
        """Try running each python script in the doc/source/code\
                directory and check that it doesn't raise an error."""
        current_dir = os.getcwd()
        os.chdir(os.path.join(self.docs_dir, "source"))

        def run_code(filename: str) -> None:
            """Run a python script in one thread."""
            with open(filename) as infile:
                # Run everything except plt.show() calls.
                with warnings.catch_warnings():
                    # When running the code asynchronously, matplotlib complains a bit
                    ignored_warnings = [
                        "Starting a Matplotlib GUI outside of the main thread",
                        ".*fetching the attribute.*Polygon.*",
                    ]
                    # This is a GeoPandas issue
                    warnings.simplefilter("error")

                    for warning_text in ignored_warnings:
                        warnings.filterwarnings("ignore", warning_text)
                    try:
                        exec(infile.read().replace("plt.show()", "plt.close()"))
                    except Exception as exception:
                        if isinstance(exception, DeprecationWarning):
                            print(exception)
                        else:
                            raise RuntimeError(f"Failed on {filename}") from exception

        filenames = [os.path.join("code", filename) for filename in os.listdir("code/") if filename.endswith(".py")]

        # Some of the doc scripts in code/ fails on Windows due to permission errors
        if (platform.system() == "Linux") or (platform.system() == "Darwin"):

            for filename in filenames:
                run_code(filename)
            """
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=int(self.n_threads) if self.n_threads is not None else None
            ) as executor:
                list(executor.map(run_code, filenames))
            """

        os.chdir(current_dir)

    def test_build(self) -> None:
        """Try building the documentation and see if it works."""

        # Ignore all warnings raised in the documentation
        # (some UserWarning are shown on purpose in certain examples, so they shouldn't make the test fail,
        # and most other warnings are for Sphinx developers, not meant to be seen by us; or we can check on RTD)
        warnings.filterwarnings("ignore")

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
