# How to contribute

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

These can all be installed with this command:
```bash
pip install prospector git+https://github.com/mypy/mypy.git pydocstyle autopep8 isort
```
Note that your text editor of choice will also need to be configured with these tools (and max character length changed).

## Test conventions
At least one test per feature (in the associated `tests/test_*.py` file) should be included in the PR, but more than one is suggested.
We use `pytest`.


## Development environment
We target Python 3 or higher for `geoutils`.
Some features may require later versions of Python (3.6+) to function correctly.

### Setup

Clone the git repo and create a conda environment
```bash
git clone https://github.com/GlacioHack/geoutils.git
cd geoutils
conda create -f environment.yml  # add '-n custom_name' if you want.
conda activate geoutils  # or any other name specified above
pip install -e .  # Install geoutils in developer mode
```
The linters and formatters mentioned above are recommended to install now.

### Running the tests
To run the entire test suite, run pytest in the current directory:
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

To validate your code automatically, install `pre-commit`:
```bash
pip install pre-commit
```
`pre-commit` is made to be installed as a "pre-commit hook" for git, so the checks have to pass before committing.
The hook is automatically installed using:
```bash
pre-commit install
```

It can also be run as a separate tool:
```bash
pre-commit run --all-files
```
