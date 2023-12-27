# PyTorchLightningTemplate

## Environment
There is a `requirements.txt` file in the root directory.
I recommend creating a virtual environment and installing the packages from the requirements file.

Conda:
- `conda create --name myenv python=3.10`
- `conda activate myenv`
- `pip install -r requirements.txt`

## Tests
Run the tests using `pytest` from the root directory.
I have had some issues sometimes, so I use `python -m pytest` instead.
You can add the following options:
- `python -m pytest tests/data` to run only the tests in the `tests/data` directory (or any other directory)
- `-s` to print the output (print statements) of the tests to the console
- `-v` to see the name of the tests that are run
- `-k "test_name"` to run only the tests that contain the string "test_name" (k stands for keyword)
