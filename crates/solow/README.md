# solow

**The comprehensive statistics and machine learning stack for Rust.** 57 focused crates. Memory safe. Pure Rust. Deterministic.

## Install

```toml
[dependencies]
solow = "0.7"
```

## Quick start

```rust
use solow::prelude::*;
use ndarray::array;

let y = array![2.5, 3.4, 4.7, 5.1, 6.5];
let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0], [1.0, 5.0]];
let res = LinearModel::ols(y, x).unwrap().fit().unwrap();
assert!(res.rsquared > 0.98);
```

## What is covered

| Group | Modules |
|---|---|
| Linear models | `regression`, `glm`, `discrete`, `robust` |
| Time series | `tsa`, `statespace`, `var`, `regime` |
| Panel and survival | `mixed`, `gee`, `duration` |
| Multivariate | `multivariate`, `cross_decomposition`, `covariance` |
| Machine learning | `svm`, `tree`, `ensemble`, `neural`, `naive_bayes`, `neighbors`, `discriminant`, `cluster` |
| Dimensionality reduction | `decomposition`, `manifold`, `kernel_approx` |
| Semi, multi, probability | `semi_supervised`, `multi`, `calibration`, `gp` |
| Model selection and metrics | `cv`, `metrics`, `pipeline`, `feature_selection` |
| Preprocessing and data | `preprocessing`, `text`, `impute`, `datasets` |
| Bayesian and formulas | `bayes`, `emplike`, `copula`, `formula`, `fit` |
| Statistics | `stats`, `nonparametric`, `gam`, `othermod` |
| Distributions and graphics | `distributions`, `viz`, `graphics`, `summary` |

Every subcrate is reachable at `solow::<module>`.

## Beyond the classical stack

Capabilities that few libraries expose as first-class modules.

- **Change-point detection.** `cusum`, `pelt` (Killick, Fearnhead, Eckley 2012), `binary_segmentation`.
- **Volatility.** `Garch11` with iterated multi-step variance forecast.
- **Extreme value analysis.** `Gev`, `Gpd` with `return_level(T)` and peaks-over-threshold fit.
- **Effect sizes.** `cohens_d`, `hedges_g`, `glass_delta`, `eta_squared`, `omega_squared`, `cliffs_delta`, `cramers_v`.
- **Meta-analysis.** `meta_fixed_effect`, `meta_random_effects` with Cochran Q, I squared, tau squared.
- **Control charts.** `ewma` and two-sided `cusum`.
- **Block bootstrap.** Moving, circular, stationary variants for time series.

## Correctness

Every deterministic estimator is cross-verified against committed golden reference fixtures. Closed-form solvers match bit-wise to `1e-10`. Iterative solvers match parameters to `1e-6` or predictions to `5e-2`. Every crate lives under `#![forbid(unsafe_code)]`.

## Full workspace

The workspace ships 57 focused crates. Depend on the umbrella for the full surface or pick individual crates for a leaner build.

```toml
[dependencies]
solow-regression = "0.7"
solow-metrics    = "0.7"
solow-cv         = "0.7"
```

See the [full workspace list](https://github.com/benovamurat/solow/blob/main/docs/book/src/crates.md) and the [documentation site](https://benovamurat.github.io/solow/) for per-module walkthroughs.

## License

BSD-3-Clause. Copyright (c) 2026, Murat Ova (Stochastic Minds).
