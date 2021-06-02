import os
import shutil
import subprocess
import sys
import warnings

from sphinx.cmd.build import main


class TestDocs:
    docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../", "doc/")

    def test_example_code(self):
        """Try running each python script in the docs/source/code\
                directory and check that it doesn't raise an error."""
        current_dir = os.getcwd()
        os.chdir(os.path.join(self.docs_dir, "source"))

        # Copy the environment and unset the DISPLAY variable to hide matplotlib plots.
        env = os.environ.copy()
        env["DISPLAY"] = ""

        for filename in os.listdir("code/"):
            if not filename.endswith(".py"):
                continue
            print(f"Running {os.path.join(os.getcwd(), 'code/', filename)}")
            subprocess.run([sys.executable, f"code/{filename}"], check=True, env=env)

        os.chdir(current_dir)

    def test_build(self):
        """Try building the docs and see if it works."""

        # Remove the build directory if it exists.
        if os.path.isdir(os.path.join(self.docs_dir, "build/")):
            shutil.rmtree(os.path.join(self.docs_dir, "build/"))

        # Copy the environment and set the SPHINXBUILD variable to call the module.
        # This is for it to work properly with GitHub Workflows
        env = os.environ.copy()
        env["SPHINXBUILD"] = f"{sys.executable} -m sphinx"

        # Run the makefile
        build_commands = ["make", "-C", self.docs_dir, "html"]
        build_commands = [
            sys.executable,
            "-m",
            "sphinx",
            os.path.join(self.docs_dir, "source"),
            os.path.join(self.docs_dir, "build"),
        ]
        main([os.path.join(self.docs_dir, "source"), os.path.join(self.docs_dir, "build")])
        """
        result = subprocess.run(
            build_commands,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            env=env
        )
        """

        """
        # Raise an error if the string "error" is in the stderr.
        if "error" in str(result.stderr).lower():
            raise RuntimeError(result.stderr)

        # If "error" is not in the stderr string but it exists, show it as a warning.
        if len(result.stderr) > 0:
            warnings.warn(result.stderr)
        """
