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
import torch
from scipy.stats import expon

from .. import est_extreme_value_index


def sim_vec_expon_ar1_sorted(
    xi: float = 2,
    phi: float = 0.5,
    num_sim: int = 365,
    seed: int = 0,
    device: str = "cuda",
) -> torch.Tensor:
    # ! scale = 1/λ = ξ
    vec = torch.from_numpy(
        expon.rvs(
            loc=0,
            scale=xi,
            size=num_sim,
            random_state=seed,
        )
    ).to(device=device)
    for i in range(1, num_sim):
        vec[i] += phi * vec[i - 1]
    return vec.exp().sort().values


def est_extreme_value_index_hill(vec: torch.tensor, k: int = None) -> float:
    # ! assuming vec is preprocessed (via pot or bm, and sorted!)
    if k is None:
        k = len(vec) // 20
    return (torch.log(vec[-k:] / vec[-k - 1]).sum() / k).item()


def est_extreme_value_index_schultze_steinebach(
    vec: torch.tensor, k: int = None
) -> float:
    # ! assuming vec is preprocessed (via pot or bm, and sorted!)
    if k is None:
        k = len(vec) // 20
    vec_j = torch.arange(1, k + 1, device=vec.device)
    tmp_vec = torch.log(len(vec) / vec_j)
    return ((tmp_vec * torch.log(vec[-vec_j])).sum() / (tmp_vec.square()).sum()).item()


def est_extreme_value_index_smith(vec: torch.tensor, q: float = None) -> float:
    # ! assuming vec is preprocessed (via pot or bm, and sorted!)
    if q is None:
        q = 0.95
    u = vec.quantile(q)
    return torch.log(vec[vec > u] / u).mean().item()


def est_extreme_value_index_meerschaert_scheffler(vec: torch.tensor) -> float:
    return (
        (vec - vec.mean()).square().sum().log() / torch.as_tensor(vec.size()).log() / 2
    ).item()


def est_EVI_benchmark(arg_in: tuple) -> dict:
    xi, phi, seed, device = arg_in
    vec = sim_vec_expon_ar1_sorted(xi=xi, phi=phi, seed=seed, device=device)
    res = {
        "evi": xi,
        "phi": phi,
        "seed": seed,
        "hill": est_extreme_value_index_hill(vec),
        "schultze_steinebach": est_extreme_value_index_schultze_steinebach(vec),
        "smith": est_extreme_value_index_smith(vec),
        "meerschaert_scheffler": est_extreme_value_index_meerschaert_scheffler(vec),
    }
    # !
    res_e2 = est_extreme_value_index(vec.cpu(), delta=1e-2, is_weighted=True)
    res["wls_emr_e2"] = res_e2.get("slope_emr")
    res["wls_mpmr_e2"] = res_e2.get("slope_mpmr")
    return res
