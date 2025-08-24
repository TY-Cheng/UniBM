import torch

from .. import est_extreme_value_index


def sim_vec_expon_ar1_sorted(
    xi: float = 2.0,
    phi: float = 0.5,
    num_sim: int = 365,
    seed: int = 0,
    device: str = "cpu",
) -> torch.Tensor:
    """Generates a SORTED random vector with a specified Extreme Value Index (xi).

    The method works by first creating an AR(1) process with innovations from an
    Exponential distribution whose scale parameter is `xi`. This process has an
    exponential tail (Gumbel domain, EVI=0). By applying an element-wise
    exponentiation (`.exp()`), this process is transformed into one with a
    Pareto-type tail (Fréchet domain), whose EVI is equal to the `xi` scale
    parameter.

    Args:
        xi (float, optional): The scale parameter for the Exponential distribution. Defaults to 2.
        phi (float, optional): The autoregressive parameter for the AR(1) process. Defaults to 0.5.
        num_sim (int, optional): The number of simulations (length of the vector). Defaults to 365.
        seed (int, optional): The random seed for reproducibility. Defaults to 0.
        device (str, optional): The device to run the simulation on. Defaults to "cpu".

    Returns:
        torch.Tensor: A sorted tensor of simulated values.
    """
    # ! scale = 1/λ = ξ
    # * AR(1) process in log-space: Y_t = phi * Y_{t-1} + innovation_t
    gen = torch.Generator(device=device).manual_seed(seed)
    vec = torch.rand(num_sim, device=device, generator=gen, dtype=torch.float64)
    vec = -torch.log1p(-vec) * xi
    for i in range(1, num_sim):
        vec[i] = vec[i] + phi * vec[i - 1]
    # * Exponentiation Z = exp(Y) gives a Pareto tail with EVI = xi; sort ascending
    return vec.exp().sort().values


def est_extreme_value_index_hill(vec: torch.tensor, k: int = None) -> float:
    # ! assuming vec is preprocessed (via pot or bm, and sorted!)
    n = vec.numel()
    if k is None:
        k = max(1, min(max(3, n // 20), n - 2))
    return (torch.log(vec[-k:] / vec[-k - 1]).mean()).item()


def est_extreme_value_index_schultze_steinebach(vec: torch.tensor, k: int = None) -> float:
    # ! assuming vec is preprocessed (via pot or bm, and sorted!)
    n = vec.numel()
    if k is None:
        k = max(1, min(max(3, n // 20), n - 1))
    vec_j = torch.arange(1, k + 1, device=vec.device)
    tmp_vec = torch.log(n / vec_j)
    return ((tmp_vec * torch.log(vec[-vec_j])).sum() / tmp_vec.square().sum()).item()


def est_extreme_value_index_smith(vec: torch.tensor, q: float = None) -> float:
    # ! assuming vec is preprocessed (via pot or bm, and sorted!)
    if q is None:
        q = 0.95
    u = vec.quantile(q)
    return torch.log(vec[vec > u] / u).mean().item()


def est_extreme_value_index_meerschaert_scheffler(vec: torch.tensor) -> float:
    return (
        (vec - vec.mean()).square().sum().log()
        / torch.as_tensor(vec.numel(), dtype=vec.dtype, device=vec.device).log()
        / 2.0
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


if __name__ == "__main__":
    vec = sim_vec_expon_ar1_sorted(device="cpu", xi=2, phi=0.5, num_sim=365, seed=0)
    vec1 = vec.clone()
    print(vec.log())
    print(est_extreme_value_index_hill(vec))
    print(est_extreme_value_index_schultze_steinebach(vec))
    print(est_extreme_value_index_smith(vec))
    print(est_extreme_value_index_meerschaert_scheffler(vec))
    assert torch.allclose(vec, vec1)
