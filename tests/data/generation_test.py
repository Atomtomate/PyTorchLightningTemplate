import os
import h5py
from src.data.generation import generate_data

def test_generate_data():
    generate_data(n=100,
                  dim=10,
                  h5path="tests/data/data/example.h5",
                  noise_params={"mean": 0.0, "std": 0.1})

    # check that file exists
    assert os.path.isfile("tests/data/data/example.h5")

    # read data
    with h5py.File("tests/data/data/example.h5", "r") as hf:
        x = hf["x"][:]
        y = hf["y"][:]
    assert x.shape == (100, 10)
    assert y.shape == (100, 2)

    # remove file
    os.remove("tests/data/data/example.h5")
