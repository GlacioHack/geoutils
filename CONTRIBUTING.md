# How to contribute

## Overview: making a contribution
For more details, see the rest of this document.

1. Fork and clone the repository.
2. Set up the development environment.
3. Create a branch for the new feature or bug fix.
4. Make your changes, ensuring they are conventions-compliant.
5. Commit, making sure to run `pre-commit` separately if not installed as git hook.
6. Push.
7. Open a Pull Request to discuss and eventually merge.
8. You can now delete your feature branch.


## Rights
The license (see LICENSE) applies to all contributions.


## Issue Conventions
When submitting bugs, please fill out the bug report template in full.

Please search existing issues, open and closed, before creating a new one.


## Git conventions
Work on features should be made on a fork of `geoutils` and submitted as a pull request (PR) to master or a relevant branch.


## Code conventions

Contributors of `geoutils` should attempt to conform to pep8 coding standards.
An exception to the standard is having a 120 max character line length (instead of 80).

Suggested linters are:
1. prospector
2. mypy (git version)
3. pydocstyle

Suggested formatters are:
1. autopep8
2. isort

## Test conventions
At least one test per feature (in the associated `tests/test_*.py` file) should be included in the PR, but more than one is suggested.
We use `pytest`.


## Development environment
We target Python 3 or higher for `geoutils`.
Some features may require later versions of Python (3.6+) to function correctly.


### Setup

Clone the git repo and create a conda environment:

```bash
git clone https://github.com/GlacioHack/geoutils.git
cd geoutils
conda env create -f environment.yml  # add '-n custom_name' if you want.
conda activate geoutils  # or any other name specified above
pip install -e .  # Install geoutils in developer mode
```

Now install the additional development dependencies:
```bash
pip install -r dev-requirements.txt
```

Finally, we also recommend installing the formatting and linting tools listed above to help with adhering to code conventions:
```bash
pip install prospector git+https://github.com/python/mypy.git pydocstyle autopep8 isort
```

Note that your text editor of choice will also need to be configured with these tools (and max character line length changed to 120).


### Running the tests
To run the entire test suite, run `pytest` in the current directory:
```bash
pytest
```

It is also recommended to try the tests from the parent directory, to validate that import statements work as they should:
```bash
cd ../  # Change to the parent directory
pytest geoutils
```

### Formatting and linting
To merge a PR in geoutils, the code has to adhere to the standards set in place.
We use a number of tools to validate contributions, triggered using `pre-commit` (see `.pre-commit-config.yaml` for the exact tools).

`pre-commit` is made to be installed as a "pre-commit hook" for git, so the checks have to pass before committing. Before committing for the first time, you need to install the hook:
```bash
pre-commit install
```

Pre-commit can also be run as a separate tool:
```bash
pre-commit run --all-files
```
