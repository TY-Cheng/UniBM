# Copyright (C) 2024- Tuoyuan Cheng, Kan Chen
#
# UniBM is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# UniBM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with UniBM. If not, see <http://www.gnu.org/licenses/>.
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import est_EVI_benchmark

file_out = Path("./out/dct_res_expon_ar1.pkl")


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
