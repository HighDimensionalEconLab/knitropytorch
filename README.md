![.github/workflows/pytest.yml](https://github.com/HighDimensionalEconLab/knitropytorch/workflows/.github/workflows/pytest.yml/badge.svg)
# knitropytorch
Pytorch support for Knitro


## Getting Started

To set up your local development environment, please use a fresh virtual environment.

Then run:

    pip install -r requirements.txt -r requirements-dev.txt

You can now import functions and classes from the module with `import knitropytorch`.

### Testing

We use `pytest` as test framework. To execute the tests, please run

    pytest

To run the tests with coverage information, please use

    python setup.py testcov

and have a look at the `htmlcov` folder, after the tests are done.

