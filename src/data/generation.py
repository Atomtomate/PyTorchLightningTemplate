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

def generate_data(n: int, dim:int, h5path: str, noise_params: Optional[dict]=None, verbose: bool=True) -> None:
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
    # chatgpt came up with this function
    for i in range(n):
        y[i, 0] = np.sin(np.sum(x[i, :])) + np.exp(np.mean(x[i, :])) * np.log(1 + np.abs(x[i, 0]))
        y[i, 1] = np.cos(np.prod(x[i, :])) / (1 + np.abs(np.sum(x[i, :]**2)))

    # add noise
    if noise_params is not None:
        y += np.random.normal(noise_params["mean"], noise_params["std"], size=y.shape)

    if verbose:
        print("x.shape", x.shape)
        print("y.shape", y.shape)

    # make sure directory exists
    os.makedirs(os.path.dirname(h5path), exist_ok=True)
    # store data in h5path
    with h5py.File(h5path, "w") as hf:
        hf.create_dataset("x", data=x)
        hf.create_dataset("y", data=y)
        if verbose:
            print("data stored in", h5path)

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
                  config.get("DATA_GENERATION", "h5path"),
                  noise_params={"mean": config.getfloat("DATA_GENERATION", "noise_mean"),
                                "std": config.getfloat("DATA_GENERATION", "noise_std")})

if __name__ == "__main__":
    main()
