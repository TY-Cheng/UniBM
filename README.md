# UniBM

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2506.14556-b31b1b.svg)](https://arxiv.org/abs/2506.14556)

`UniBM` is a Python toolkit for extreme risk modeling and statistical analysis, with a focus on methods like sub-sampling block maxima as exampled in [`vignette.ipynb`](./vignette.ipynb).
The functions in this repository are designed to accompany the methods detailed in our research paper. For in-depth explanations and applications, users are strongly encouraged to consult the paper.

All bug reports and feature requests are welcomed.

The scripts are licensed under the [MIT License](./LICENSE).

## Citation

If you use `unibm` in your work, please cite:

> Cheng, T., Peng, X., Choiruddin, A., He, X., & Chen, K. (2025). Environmental extreme risk modeling via sub-sampling block maxima. arXiv preprint arXiv:2506.14556.

```latex
@article{cheng2025environmental,
title={Environmental extreme risk modeling via sub-sampling block maxima},
author={Cheng, Tuoyuan and Peng, Xiao and Choiruddin, Achmad and He, Xiaogang and Chen, Kan},
journal={arXiv preprint arXiv:2506.14556},
year={2025}
}
```

## (Recommended) [uv](https://docs.astral.sh/uv/getting-started/) for Dependency Management and Packaging

After `git clone https://github.com/TY-Cheng/UniBM.git`, `cd` into the project root where [`pyproject.toml`](./pyproject.toml) exists,

```bash
# From inside the project root folder
# Sync dependencies with CPU support (default)
uv sync --extra cpu
```

## Functions

`unibm`

> `cdf_func_kernel` non parametric cdf estimator, by kernel smoothing
>
> `est_tail_dep_coeff` pairwise tail dependence coefficient estimator
>
> `est_extremal_index_reciprocal` extremal index (EI) estimator, by reciprocal of the mean of the block maxima;
>
> `viz_eir` chart results from `est_extremal_index_reciprocal(is_retn_vec=True)`
>
> `est_extreme_value_index` extreme value index (EVI) estimator, by MPMR & EMR
>
> `viz_evi_reg` chart results from `est_extreme_value_index(is_retn_vec=True)`

## License

This project is released under the [MIT License](./LICENSE) (© 2024- Tuoyuan Cheng, Kan Chen).
