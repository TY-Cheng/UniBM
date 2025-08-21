import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import est_EVI_benchmark

file_out = Path("./out/dct_res_expon_ar1.pkl")
file_out.parent.mkdir(parents=True, exist_ok=True)


def generate_parameters():
    """Generate parameters for the benchmark."""
    lst_xi = np.concatenate(
        [
            np.arange(0.1, 1, 0.1),
            np.arange(1, 11, 1),
        ]
    )
    for xi in lst_xi:
        for phi in [0, 0.1, 0.5, 0.9]:
            # NOTE
            for seed in range(666):
                for device in ["cpu"]:
                    yield xi.item(), phi, seed, device


def main(file: str):
    """Main function to run the benchmark and save results."""
    try:
        with Pool() as pool:
            results = pool.map(func=est_EVI_benchmark, iterable=generate_parameters())
        with open(file=file, mode="wb") as f:
            pickle.dump(obj=results, file=f)
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main(file=file_out)
