"""This script generates data for the example.
run with `python -m src.data.generation --config <path to config file>`
For help run `python -m src.data.generation --help`
"""
import numpy as np
from scipy.stats import multivariate_normal
import h5py
from ..utils.utils import read_config
import argparse
import textwrap
import os
from typing import Optional

def generate_data(n: int, dim:int, data_output_dir: str, noise_params: Optional[dict]=None, verbose: bool=True) -> None:
    """generate n events and store them in h5path
    :param n: number of events to generate
    :param dim: dimension of the data
    :param h5path: path to store the data

    :return: None

    Here we generate x in [-10,10]^dim and y = f(x) + noise.
    The range is not [0,1]^dim so we later have to rescale the data, which one usually needs to do.

    x has shape (n, dim)
    y has shape (n, 2)

    """
    x = np.random.rand(n, dim) * 20 - 10

    y = np.zeros((n, 2))
    for i in range(n):
        y[i, 0] = np.sin(np.sum(x[i, :]))
        y[i, 1] = 2*np.cos(x[i, 1]) + np.sin(x[i, 0]*3)

    # add noise
    if noise_params is not None: y += np.random.normal(noise_params["mean"], noise_params["std"], size=y.shape)

    if verbose:
        print("x.shape", x.shape)
        print("y.shape", y.shape)

    # make sure directory exists
    os.makedirs(data_output_dir, exist_ok=True)
    # store data in h5path
    h5path = os.path.join(data_output_dir, "data_full.h5")
    with h5py.File(h5path, "w") as hf:
        hf.create_dataset("x", data=x)
        hf.create_dataset("y", data=y)
        if verbose:
            print("data stored in", h5path)

def train_test_split(data_output_dir: str, train_fraction: float=0.8, val_fraction: float=0.1, test_fraction: float=0.1, verbose: bool=True) -> None:
    """Split data from h5path into train, val and test and store the data in train_path, val_path and test_path respectively.
    :param h5path: path to the data
    :param train_path: path to store the train data
    :param val_path: path to store the val data
    :param test_path: path to store the test data
    :param train_fraction: fraction of the data to use for training
    :param val_fraction: fraction of the data to use for validation
    :param test_fraction: fraction of the data to use for testing
    """
    # read data
    h5path = os.path.join(data_output_dir, "data_full.h5")
    with h5py.File(h5path, "r") as hf:
        x = hf["x"][:]
        y = hf["y"][:]
    # split data
    n = x.shape[0]
    n_train = int(n * train_fraction)
    n_val = int(n * val_fraction)
    n_test = n - n_train - n_val
    if verbose:
        print("n_train", n_train)
        print("n_val", n_val)
        print("n_test", n_test)
    # shuffle data
    perm = np.random.permutation(n)
    x = x[perm]
    y = y[perm]
    # split data
    x_train = x[:n_train]
    y_train = y[:n_train]
    x_val = x[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]
    x_test = x[n_train+n_val:]
    y_test = y[n_train+n_val:]
    # store data
    train_path = os.path.join(data_output_dir, "data_train.h5")
    val_path = os.path.join(data_output_dir, "data_val.h5")
    test_path = os.path.join(data_output_dir, "data_test.h5")
    with h5py.File(train_path, "w") as hf:
        hf.create_dataset("x", data=x_train)
        hf.create_dataset("y", data=y_train)
    with h5py.File(val_path, "w") as hf:
        hf.create_dataset("x", data=x_val)
        hf.create_dataset("y", data=y_val)
    with h5py.File(test_path, "w") as hf:
        hf.create_dataset("x", data=x_test)
        hf.create_dataset("y", data=y_test)
    if verbose:
        print("data stored in")
        print(train_path)
        print(val_path)
        print(test_path)

def main() -> None:
    """Generate data with params from config file and export as h5"""
    parser = argparse.ArgumentParser(description="Generate example data",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=textwrap.dedent('''\
                                    Needs a configuration (`--config <path_to_config_file>`) file, typically called something like `config.ini` and looking like this (adjust parameters):
                                        [DATA_GENERATION]
                                        n = 10000
                                        dim = 10
                                        h5path = data/example.h5
                                        noise_mean = 0.0
                                        noise_std = 0.1
                                    ''')
                                     )
    parser.add_argument('--config', type=str, help='Path to the configuration file', required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    generate_data(config.getint("DATA_GENERATION", "n"),
                  config.getint("DATA_GENERATION", "dim"),
                  config.get("DATA_GENERATION", "data_output_dir"),
                  noise_params={"mean": config.getfloat("DATA_GENERATION", "noise_mean"),
                                "std": config.getfloat("DATA_GENERATION", "noise_std")})
    # split data
    train_test_split(config.get("DATA_GENERATION", "data_output_dir"),
                     train_fraction=config.getfloat("DATA_GENERATION", "train_fraction"),
                     val_fraction=config.getfloat("DATA_GENERATION", "val_fraction"),
                     test_fraction=config.getfloat("DATA_GENERATION", "test_fraction"))

if __name__ == "__main__":
    main()
