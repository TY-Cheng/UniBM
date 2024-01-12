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


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import euler_gamma
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.special import digamma, loggamma, ndtr
from statsmodels.api import OLS, WLS


def cdf_func_kernel(vec: np.array, is_scott: bool = True) -> callable:
    """non parametric cdf estimator, by kernel smoothing

    :param vec: raw observations
    :type vec: np.array
    :param is_scott: bandwidth by Scott 1992, defaults to True
    :type is_scott: bool, optional
    :return: cdf estimator, as a function
    :rtype: callable
    """
    if is_scott:
        # * bandwidth by Scott 1992
        band_width = (
            np.diff(np.nanquantile(a=vec, q=(0.25, 0.75), method="median_unbiased"))
            / 1.349
        )
        band_width = 1.059 * min(np.nanstd(vec), band_width) * vec.size ** (-0.2)
    else:
        band_width = (
            np.nanstd(vec) * 0.6973425390765554 * (vec.size) ** (-0.1111111111111111)
        )

    @np.vectorize
    def func_cdf(q: np.array) -> np.array:
        return np.nanmean(ndtr((q - vec) / band_width))

    return func_cdf


def est_tail_dep_coeff(
    vec_1: np.array,
    vec_2: np.array,
    is_FF: bool = True,
    is_upper: bool = True,
) -> float:
    """pairwise tail dependence coefficient (λ) estimator

    :param vec_1: observations of variable 1
    :type vec_1: np.array
    :param vec_2: observations of variable 2
    :type vec_2: np.array
    :param is_FF: FF 2013 or CFG 2005, defaults to True
    :type is_FF: bool, optional
    :param is_upper: upper or lower tail, defaults to True
    :type is_upper: bool, optional
    :return: _description_
    :rtype: float

    Ferreira, M.S., 2013. Nonparametric estimation of the tail-dependence coefficient;
    Frahm, G., Junker, M. and Schmidt, R., 2005. Estimating the tail-dependence coefficient: properties and pitfalls. Insurance: mathematics and Economics, 37(1), pp.80-100.
    """

    # marginal transform to copula
    idx = np.isfinite(vec_1) & np.isfinite(vec_2)
    vec_u_1 = vec_1[idx]
    vec_u_1 = cdf_func_kernel(vec=vec_u_1)(vec_u_1)
    vec_u_2 = vec_2[idx]
    vec_u_2 = cdf_func_kernel(vec=vec_u_2)(vec_u_2)
    if not is_upper:
        # rotate the bivariate copula by 180 deg
        vec_u_1, vec_u_2 = 1 - vec_u_1, 1 - vec_u_2
    vec_u_max = np.maximum(vec_u_1, vec_u_2)
    if is_FF:
        return 3 - 1 / (1 - vec_u_max.mean())
    else:
        return 2 - 2 * np.exp(
            (
                0.5
                * (
                    np.log(np.log(np.reciprocal(vec_u_1)))
                    + np.log(np.log(np.reciprocal(vec_u_2)))
                )
                - np.log(2 * np.log(np.reciprocal(vec_u_max)))
            ).mean()
        )


def est_extremal_index_reciprocal(
    srs: pd.Series,
    num_step: int = None,
    size_start: int = None,
    size_stop: int = None,
    is_geom: bool = None,
    is_retn_vec: bool = False,
) -> dict:
    """extremal index reciprocal (1/θ) estimator, by reciprocal of the mean of the block maxima;

    :param srs: a time series of observations
    :type srs: pd.Series
    :param num_step: number of different block sizes, defaults to None
    :type num_step: int, optional
    :param size_start: minimum block size, defaults to None
    :type size_start: int, optional
    :param size_stop: maximum block size, defaults to None
    :type size_stop: int, optional
    :param is_geom: arrange block size geometrically or arithmetically, defaults to None
    :type is_geom: bool, optional
    :param is_retn_vec: keep result vectors for further charting, defaults to False
    :type is_retn_vec: bool, optional
    :return: _description_
    :rtype: dict

    * Northrop, P.J., 2015. An efficient semiparametric maxima estimator of the extremal index.
    Extremes, 18, pp.585-603.
    * Betina Berghaus. Axel Bücher. Weak convergence of a pseudo maximum likelihood estimator for the extremal index.
    Ann. Statist. 46 (5) 2307 - 2335, October 2018
    * Ferreira, M., 2023. Clustering of extreme values: estimation and application.
    AStA Advances in Statistical Analysis, pp.1-25.
    """
    # ! dnt truncate, dnt shift, dnt sort (keep temporal consecutive)
    # ! srs values should be strictly greater than 0 !
    vec = np.log1p(srs.values)
    vec -= np.nanmin(vec)

    # ! vec_size: a vector of sample sizes
    N1 = vec.size + 1
    if size_start is None:
        size_start = int(np.max([10, np.exp(np.log(N1) * 0.2857142857142857)]))
    if size_stop is None:
        size_stop = int(np.exp(np.log(N1) * 0.6666666666666666))
    if is_geom is None:
        is_geom = False if N1 <= 5e3 else True
    if is_geom:
        if num_step is None:
            num_step = int(min(128, np.log(size_stop) * 100))
        vec_size = np.unique(
            (
                np.geomspace(
                    start=size_start, stop=size_stop, num=num_step, endpoint=False
                )
            ).astype(int)
        )
    else:
        if num_step == None:
            num_step = size_stop if size_stop < 5e3 else size_stop // 2
        vec_size = np.arange(
            start=size_start, stop=size_stop, step=size_stop // num_step
        ).astype(int)

    dct_res = {"vec_l_size": np.log(vec_size)}

    # * the raw Poisson seq, by Northrop
    srs_pois = -pd.Series(np.log(cdf_func_kernel(vec=vec)(vec)), index=srs.index)

    @np.vectorize
    def _get_eir_sd(size: int):
        vec_eir = srs_pois.rolling(
            window=size, min_periods=size, step=int(np.ceil(size**0.3))
        ).apply(lambda s: size * s.min())
        return vec_eir.mean(), vec_eir.std()

    vec_eir, vec_sd = _get_eir_sd(vec_size)
    vec_eir = vec_eir.clip(min=1)
    _ = vec_sd.argmax()
    dct_res["size_Northrop"] = vec_size[_]
    dct_res["eir_Northrop"] = vec_eir[_]
    if is_retn_vec:
        dct_res["vec_eir_Northrop"] = vec_eir
        dct_res["vec_sd_Northrop"] = vec_sd

    # * the raw Poisson seq, by BB
    srs_pois = pd.Series(1 - cdf_func_kernel(vec=vec)(vec), index=srs.index)

    @np.vectorize
    def _get_eir_sd(size: int):
        # ! notice the min here
        vec_eir = srs_pois.rolling(
            window=size, min_periods=size, step=int(np.ceil(size**0.3))
        ).apply(lambda s: size * s.min())
        return vec_eir.mean(), vec_eir.std()

    vec_eir, vec_sd = _get_eir_sd(vec_size)
    # ! bias adjustment
    # Ferreira, M., 2023. Measuring Extremal Clustering in Time Series. Engineering Proceedings, 39(1), p.64.
    vec_eir = np.reciprocal(np.reciprocal(vec_eir) - 1 / vec_size)
    vec_eir = vec_eir.clip(min=1)
    _ = vec_sd.argmax()
    dct_res["size_BB"] = vec_size[_]
    dct_res["eir_BB"] = vec_eir[_]
    if is_retn_vec:
        dct_res["vec_eir_BB"] = vec_eir
        dct_res["vec_sd_BB"] = vec_sd

    return dct_res


def viz_eir(
    dct_res: dict,
    file_path: Path = None,
    fig_title: str = None,
    is_save: bool = False,
    xlabel: str = "log (block size)",
    ylabel: str = "Z",
) -> None:
    """chart results from est_extremal_index_reciprocal(is_retn_vec=True)

    :param dct_res: dictionary with keys: vec_l_size, vec_eir_Northrop, vec_eir_BB, vec_sd_Northrop, vec_sd_BB
    :type dct_res: dict
    :param file_path: _description_, defaults to None
    :type file_path: Path, optional
    :param fig_title: _description_, defaults to None
    :type fig_title: str, optional
    :param is_save: _description_, defaults to False
    :type is_save: bool, optional
    :param xlabel: _description_, defaults to None
    :type xlabel: str, optional
    :param ylabel: _description_, defaults to None
    :type ylabel: str, optional
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=173)
    tmp_df = pd.DataFrame(
        {
            k: dct_res[k]
            for k in dct_res.keys()
            if k
            in [
                "vec_l_size",
                "vec_eir_Northrop",
                "vec_eir_BB",
                "vec_sd_Northrop",
                "vec_sd_BB",
            ]
        }
    )
    tmp_df.plot(
        ax=ax,
        kind="scatter",
        x="vec_l_size",
        y="vec_eir_Northrop",
        label=f"eir_Northrop, {dct_res['eir_Northrop']:.4f}",
        color="tab:cyan",
        alpha=0.7,
        s=3,
    )
    tmp_df.plot(
        ax=ax,
        kind="scatter",
        x="vec_l_size",
        y="vec_sd_Northrop",
        label="sd_Northrop",
        color="tab:blue",
        alpha=0.7,
        s=2.5,
    )
    tmp_df.plot(
        ax=ax,
        kind="scatter",
        x="vec_l_size",
        y="vec_eir_BB",
        label=f"eir_BB, {dct_res['eir_BB']:.4f}",
        color="tab:red",
        alpha=0.7,
        s=3,
    )
    tmp_df.plot(
        ax=ax,
        kind="scatter",
        x="vec_l_size",
        y="vec_sd_BB",
        label="sd_BB",
        color="tab:purple",
        alpha=0.7,
        s=2.5,
    )
    _ = tmp_df.columns.difference(["vec_l_size"])
    _min, _max = tmp_df[_].min().min(), tmp_df[_].max().max()
    ax.vlines(
        x=np.log(dct_res["size_Northrop"]),
        ymin=_min - 0.1,
        ymax=_max + 0.1,
        colors="tab:blue",
        linestyles=":",
        lw=0.9,
    )
    ax.vlines(
        x=np.log(dct_res["size_BB"]),
        ymin=_min - 0.1,
        ymax=_max + 0.1,
        colors="tab:purple",
        linestyles=":",
        lw=0.9,
    )
    ax.set_xlabel(xlabel=xlabel)
    ax.set_ylabel(ylabel=ylabel)
    ax.set_title(fig_title)
    ax.legend()
    ax.grid()
    fig.tight_layout()
    if is_save:
        fig.savefig(fname=file_path)
        plt.close()
    pass


def est_extreme_value_index(
    vec: np.array,
    delta: float = 0.17,
    num_step: int = None,
    is_frechet: bool = True,
    is_geom: bool = None,
    is_deming: bool = False,
    is_retn_vec: bool = False,
    cov_type: str = "hc0",
    use_t: bool = True,
) -> dict:
    """extreme value index (ξ) estimator, by MPMR & EMR

    :param vec: a vector of observations
    :type vec: np.array
    :param delta: hyperparameter to restrict search along std.dev., defaults to 0.17
    :type delta: float, optional
    :param num_step: number of different block sizes, defaults to None
    :type num_step: int, optional
    :param is_frechet: observations have dist in maximum domain of attraction of Frechet or Gumbel, defaults to True
    :type is_frechet: bool, optional
    :param is_geom: arrange block size geometrically or arithmetically, defaults to False
    :type is_geom: bool, optional
    :param is_deming: regression by Deming or least square, defaults to False
    :type is_deming: bool, optional
    :param is_retn_vec: keep result vectors for further charting, defaults to False
    :type is_retn_vec: bool, optional
    :param cov_type: _description_, defaults to "hc0"
    :type cov_type: str, optional
    :param use_t: _description_, defaults to True
    :type use_t: bool, optional
    :return: _description_
    :rtype: dict
    """
    # ! the support of vec must strictly greater than 0
    # vec_x is sorted
    vec_x = np.sort(vec[vec > 0 & np.isfinite(vec)])
    if is_frechet:
        # vec (pareto, (A, 1/ξ)) -> vec_x (exponential, (0, 1/ξ))
        # from Frechet MDA to Gumbel MDA
        vec_x = np.log(vec_x)
    _ = vec_x[0]
    vec_x -= _
    vec_x = vec_x[vec_x > 0]

    # * vec_size: a vector of re-sample (block) sizes to estimate EMR & MPMR
    # * restrict upper lmt to exp(2/3*log(N1)) around sqrt(N1)
    # Oorschot, J., Segers, J. and Zhou, C., 2023. Tail inference using extreme U-statistics. Electronic Journal of Statistics, 17(1), pp.1113-1159.
    N1 = vec_x.size + 1
    size_stop = int(np.exp(np.log(N1) * 0.8646647167633873))
    if is_deming:
        is_geom = True
    if is_geom is None:
        is_geom = False if N1 <= 1e4 else True
    if is_geom:
        if num_step is None:
            num_step = int(min(128, np.log(size_stop) * 100))
        vec_size = np.unique(
            (
                np.geomspace(start=1, stop=size_stop, num=num_step, endpoint=False)
            ).astype(int)
        )
    else:
        if num_step == None:
            num_step = size_stop if size_stop < 5e3 else size_stop // 2
        vec_size = np.arange(start=1, stop=size_stop, step=size_stop // num_step)

    dct_res = {}
    dct_res["is_frechet"] = is_frechet
    dct_res["is_geom"] = is_geom
    dct_res["is_deming"] = is_deming
    if is_retn_vec:
        dct_res["log_x_min"] = _

    @np.vectorize
    def _get_mpmr_sd(size: int):
        # * emr is a 're-weighted' average focusing on tails
        # ! size start from 1
        vec_i = np.arange(start=size, stop=N1, step=1)
        vec_pmf = size * np.exp(
            loggamma(N1 - size)
            - loggamma(N1)
            + loggamma(vec_i)
            - loggamma(vec_i - size + 1)
        )
        # those with larger value (or rank)
        vec_up = vec_x[(size - 1) :]
        # 1st order raw moment
        emr = vec_pmf @ vec_up
        # 2nd order central moment
        sd = np.sqrt((vec_pmf @ np.square(vec_up) - emr**2).clip(min=0))
        # * last in first out (LIFO), log1p to accelerate the mode estimator (mean-shift)
        # ! log1p to make X more Gaussian while keeping X positive
        vec_up = np.log1p(vec_x[(size - 1) :])
        # bandwidth by Scott 1992
        band_width = (
            np.diff(np.quantile(a=vec_up, q=(0.25, 0.75), method="median_unbiased"))
            / 1.349
        )
        band_width = 1.059 * min(vec_up.std(), band_width) * vec_up.size ** (-0.2)
        mode_x_0 = vec_up.mean()
        for _ in range(1000):
            mode_x_1 = mode_x_0
            vec_kernel = np.exp(-np.square(vec_up - mode_x_1) / band_width).clip(
                min=0, max=None
            )
            mode_x_0 = ((vec_kernel * vec_up) @ vec_pmf) / (vec_kernel @ vec_pmf)
            if np.isclose(a=mode_x_0, b=mode_x_1):
                break
        # * LIFO
        mode_x_0 = np.expm1(mode_x_0)
        return mode_x_0, emr, sd

    vec_mpmr, vec_emr, vec_sd = _get_mpmr_sd(vec_size)
    vec_l_size = np.log(vec_size)

    if is_retn_vec:
        # return the raw vec; otherwise later we truncate for regression
        dct_res["vec_mpmr"] = vec_mpmr
        dct_res["vec_emr"] = vec_emr
        dct_res["vec_sd"] = vec_sd
        dct_res["vec_l_size"] = vec_l_size

    # filter out numerical unstable points
    idx = (
        (vec_sd > 0)
        & (vec_mpmr > 0)
        & (vec_emr > 0)
        & (vec_l_size >= (0.1353352832366127 * np.log(N1)) - np.log1p(-delta))
        & (vec_l_size <= np.log(size_stop) - np.log1p(delta))
    )
    vec_mpmr, vec_emr, vec_l_size, vec_size, vec_sd = (
        vec_mpmr[idx],
        vec_emr[idx],
        vec_l_size[idx],
        vec_size[idx],
        vec_sd[idx],
    )
    l_size_min, l_size_max = vec_l_size[0], vec_l_size[-1]

    if is_deming:
        # * vec of harmonic number, for emr
        vec_hnum = digamma(vec_size + 1) + euler_gamma
        # * Deming
        delta = np.pi**2 / 6
        xm_mpmr, ym_mpmr, xm_emr, ym_emr = (
            vec_l_size.mean(),
            vec_mpmr.mean(),
            vec_hnum.mean(),
            vec_emr.mean(),
        )
        x_c_mpmr, y_c_mpmr, x_c_emr, y_c_emr = (
            vec_l_size - xm_mpmr,
            vec_mpmr - ym_mpmr,
            vec_hnum - xm_emr,
            vec_emr - ym_emr,
        )
        s_xx_mpmr, s_yy_mpmr, s_xx_emr, s_yy_emr = (
            np.square(x_c_mpmr).mean(),
            np.square(y_c_mpmr).mean(),
            np.square(x_c_emr).mean(),
            np.square(y_c_emr).mean(),
        )
        s_xy_mpmr, s_xy_emr = (x_c_mpmr * y_c_mpmr).mean(), (x_c_emr * y_c_emr).mean()
        ymdx_mpmr, ymdx_emr = (
            s_yy_mpmr - delta * s_xx_mpmr,
            s_yy_emr - delta * s_xx_emr,
        )
        slope_mpmr, slope_emr = (
            (ymdx_mpmr + np.sqrt(ymdx_mpmr**2 + 4 * delta * s_xy_mpmr**2))
            / (2 * s_xy_mpmr),
            (ymdx_emr + np.sqrt(ymdx_emr**2 + 4 * delta * s_xy_emr**2))
            / (2 * s_xy_emr),
        )
        intercept_mpmr, intercept_emr = (
            ym_mpmr - slope_mpmr * xm_mpmr,
            ym_emr - slope_emr * xm_emr,
        )
        dct_res["slope_mpmr"] = slope_mpmr
        dct_res["slope_emr"] = slope_emr
        dct_res["intercept_mpmr"] = intercept_mpmr
        dct_res["intercept_emr"] = intercept_emr
        dct_res["mpmr_min"] = vec_mpmr[0]
        dct_res["emr_min"] = vec_emr[0]
        dct_res["size_min"] = vec_size[0]
        dct_res["size_max"] = vec_size[-1]
        return dct_res

    else:
        # locate a region of sd plateau, by minimizing square of gradient
        # record as l_size_min & l_size_max
        spl_sd = CubicSpline(x=vec_l_size, y=vec_sd)
        spl_sd_d1 = spl_sd.derivative(1)

        def objfun(l_size) -> float:
            return spl_sd_d1(l_size) ** 2

        # * locate the reference level of std.dev., by minimizing the above objfun
        sd_ref = spl_sd(
            minimize(
                fun=objfun,
                x0=(l_size_max / 2,),
                bounds=((l_size_max / 4, l_size_max),),
                method="Nelder-Mead",
            ).x
        )[0]
        # * locate the max & min log(sample size); may have multiple roots to select
        _ = spl_sd.solve(sd_ref * (1 - delta))
        _ = _[(_ >= l_size_min) & (_ <= l_size_max)]
        if _.size > 0:
            l_size_max = np.max(_)
        _ = spl_sd.solve(sd_ref * (1 + delta))
        _ = _[(_ >= l_size_min) & (_ <= l_size_max)]
        if _.size > 0:
            l_size_min = np.min(_)
        idx = (vec_l_size >= l_size_min) & (vec_l_size <= l_size_max)
        vec_mpmr, vec_emr, vec_l_size, vec_size = (
            vec_mpmr[idx],
            vec_emr[idx],
            vec_l_size[idx],
            vec_size[idx],
        )
        vec_hnum = digamma(vec_size + 1) + euler_gamma
        # mpmr
        X = np.hstack(
            (np.ones_like(vec_l_size)[:, np.newaxis], vec_l_size[:, np.newaxis])
        )
        if X.size > 2:
            if is_geom:
                mdl = OLS(endog=vec_mpmr, exog=X, hasconst=True).fit(
                    method="pinv", cov_type=cov_type, use_t=use_t
                )
            else:
                mdl = WLS(
                    endog=vec_mpmr, exog=X, weights=1 / vec_size, hasconst=True
                ).fit(method="pinv", cov_type=cov_type, use_t=use_t)
            intercept_mpmr, slope_mpmr = mdl.params
        else:
            intercept_mpmr, slope_mpmr = None, None
        # emr
        X = np.hstack((np.ones_like(vec_hnum)[:, np.newaxis], vec_hnum[:, np.newaxis]))
        if X.size > 2:
            if is_geom:
                mdl = OLS(endog=vec_emr, exog=X, hasconst=True).fit(
                    method="pinv", cov_type=cov_type, use_t=use_t
                )
            else:
                mdl = WLS(
                    endog=vec_emr, exog=X, weights=1 / (1 + vec_size), hasconst=True
                ).fit(method="pinv", cov_type=cov_type, use_t=use_t)

            intercept_emr, slope_emr = mdl.params
        else:
            intercept_emr, slope_emr = None, None

        dct_res["slope_mpmr"] = slope_mpmr
        dct_res["slope_emr"] = slope_emr
        dct_res["intercept_mpmr"] = intercept_mpmr
        dct_res["intercept_emr"] = intercept_emr
        dct_res["mpmr_min"] = vec_mpmr[0]
        dct_res["emr_min"] = vec_emr[0]
        dct_res["size_min"] = vec_size[0]
        dct_res["size_max"] = vec_size[-1]
        dct_res["sd_ref"] = sd_ref
        return dct_res


def viz_evi_reg(
    dct_res: dict,
    file_path: Path = None,
    fig_title: str = None,
    is_save: bool = False,
    xlabel: str = "log (block size)",
    ylabel: str = None,
) -> None:
    """chart results from est_extreme_value_index(is_retn_vec=True)

    :param dct_res: dictionary with keys: vec_l_size, vec_mpmr, vec_emr, vec_sd, slope_mpmr, slope_emr, intercept_mpmr, intercept_emr, mpmr_min, emr_min, size_min, size_max, sd_ref
    :type dct_res: dict
    :param file_path: _description_, defaults to None
    :type file_path: Path, optional
    :param fig_title: _description_, defaults to None
    :type fig_title: str, optional
    :param is_save: _description_, defaults to False
    :type is_save: bool, optional
    :param xlabel: _description_, defaults to None
    :type xlabel: str, optional
    :param ylabel: _description_, defaults to None
    :type ylabel: str, optional
    """
    if ylabel is None:
        ylabel = "log (block maxima)" if dct_res["is_frechet"] else "block maxima"
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=173)
    tmp_df = pd.DataFrame(
        {
            k: dct_res[k]
            for k in dct_res.keys()
            if k in ["vec_mpmr", "vec_emr", "vec_sd", "vec_l_size"]
        },
        index=dct_res["vec_l_size"],
    )
    _ = dct_res["log_x_min"]
    tmp_df["vec_size"] = np.exp(tmp_df.index)
    tmp_df["vec_hnum"] = digamma(tmp_df["vec_size"] + 1) + euler_gamma
    tmp_df["vec_mpmr_fit"] = (
        tmp_df["vec_l_size"] * dct_res["slope_mpmr"] + dct_res["intercept_mpmr"]
    )
    tmp_df["vec_mpmr_fit"] += _
    tmp_df["vec_mpmr"] += _
    tmp_df["vec_emr_fit"] = (
        tmp_df["vec_hnum"] * dct_res["slope_emr"] + dct_res["intercept_emr"]
    )
    tmp_df["vec_emr_fit"] += _
    tmp_df["vec_emr"] += _
    #
    tmp_df.plot(
        ax=ax,
        y=["vec_mpmr_fit", "vec_emr_fit"],
        style=["-.", "-."],
        label=[
            f"mpmr_fit, slope={dct_res['slope_mpmr']:.4f}",
            f"emr_fit, slope={dct_res['slope_emr']:.4f}",
        ],
        alpha=0.5,
        color=["tab:cyan", "tab:red"],
    )
    ax.scatter(
        x=tmp_df["vec_l_size"],
        y=tmp_df["vec_mpmr"],
        s=3,
        alpha=0.7,
        color="tab:blue",
        label="mpmr_est",
    )
    ax.scatter(
        x=tmp_df["vec_l_size"],
        y=tmp_df["vec_emr"],
        s=3,
        alpha=0.7,
        color="tab:purple",
        label="emr_est",
    )
    ax.scatter(
        x=tmp_df["vec_l_size"],
        y=tmp_df["vec_sd"],
        s=3,
        alpha=0.7,
        color="tab:green",
        label="std.dev.",
    )
    ax.vlines(
        x=np.log([dct_res["size_min"], dct_res["size_max"]]),
        ymin=np.min([tmp_df["vec_emr"].iloc[0], tmp_df["vec_sd"].min()]) - 0.1,
        ymax=np.max([tmp_df["vec_emr_fit"].iloc[-1], tmp_df["vec_sd"].max()]) + 0.1,
        colors="k",
        linestyles=":",
        lw=1.1,
    )
    if not dct_res["is_deming"]:
        ax.hlines(
            y=dct_res["sd_ref"],
            xmin=tmp_df["vec_l_size"].iloc[0] - 0.1,
            xmax=tmp_df["vec_l_size"].iloc[-1] + 0.1,
            colors="k",
            linestyles=":",
            lw=0.9,
        )
    ax.set_xlabel(xlabel=xlabel)
    ax.set_ylabel(ylabel=ylabel)
    ax.set_title(fig_title)
    ax.legend()
    ax.grid()
    fig.tight_layout()
    if is_save:
        fig.savefig(fname=file_path)
        plt.close()
    pass
