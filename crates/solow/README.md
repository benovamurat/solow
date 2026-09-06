# solow

**The comprehensive statistics and machine learning stack for Rust.** 54 focused crates. Memory safe. Deterministic. Pure Rust.

The umbrella `solow` crate re-exports the full public API of the workspace behind a single ergonomic `prelude`. Depend on it once and you have regression, generalized linear models, discrete choice, time series, state space, survival analysis, mixed effects, Bayesian inference, clustering, tree ensembles, SVM, neural networks, kernel methods, dimensionality reduction, and change-point detection.

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

## Why Solow

- **Memory safe.** Every crate lives under `#![forbid(unsafe_code)]`.
- **Deterministic.** Every stochastic estimator uses a portable MMIX-LCG PRNG. A fixed seed reproduces bit-identical fits across runs and platforms.
- **Cross-verified.** Every deterministic estimator is checked against committed golden reference fixtures. Closed-form solvers agree bit-wise to `1e-10`.
- **Pure Rust.** No system LAPACK. No BLAS. No Python runtime. No C beyond libc.
- **Single binary.** Deploys anywhere Rust runs, including WebAssembly.
- **54 focused crates.** Depend on the umbrella for the full surface, or pick individual crates for a leaner build.

## What is covered

Every subcrate is reachable at `solow::<module>`.

| Group | Modules | Highlights |
|---|---|---|
| **Linear models** | `regression`, `glm`, `discrete`, `robust` | OLS, WLS, GLS, Ridge, Lasso, ElasticNet, LARS, Huber, quantile, Bayesian, GLM (Gaussian, Poisson, Binomial, Gamma, Tweedie), logit, probit, ordered, zero-inflated |
| **Time series** | `tsa`, `statespace`, `var`, `regime` | ARMA, SARIMAX, Kalman, VAR / VECM / SVAR, Markov switching, Holt-Winters, GARCH(1,1), STL, PELT change-point |
| **Panel and survival** | `mixed`, `gee`, `duration` | REML mixed effects, GEE, Kaplan-Meier, Nelson-Aalen, Cox PH |
| **Multivariate** | `multivariate`, `cross_decomposition`, `covariance` | PCA, MANOVA, CCA, PLS, Ledoit-Wolf, OAS, MinCovDet, graphical lasso |
| **Machine learning** | `svm`, `tree`, `ensemble`, `neural`, `naive_bayes`, `neighbors`, `discriminant`, `cluster` | SVC, SVR, random forests, gradient boosting, MLP, RBM, k-NN, LDA/QDA, k-means, DBSCAN, HDBSCAN, GMM |
| **Dimensionality reduction** | `decomposition`, `manifold`, `kernel_approx` | PCA, KernelPCA, NMF, ICA, LDA, t-SNE, MDS, Isomap, LLE, RBF sampler, Nyström |
| **Semi, multi, probability** | `semi_supervised`, `multi`, `calibration`, `gp` | Label propagation, one-vs-rest, chain, Platt scaling, isotonic, Gaussian processes |
| **Model selection and metrics** | `cv`, `metrics`, `pipeline`, `feature_selection` | K-fold, purged K-fold, grid search, halving, 100+ metrics, calibration, conformal, RFE |
| **Preprocessing and data** | `preprocessing`, `text`, `impute`, `datasets` | Scalers, encoders, power / quantile transforms, TF-IDF, MICE, make_* / load_* helpers |
| **Bayesian and formulas** | `bayes`, `emplike`, `copula`, `formula`, `fit` | Variational Bayes, empirical likelihood, copulas, R-style formulas |
| **Statistics** | `stats`, `nonparametric`, `gam`, `othermod` | 40+ tests, LOWESS, KDE, GAM, meta-analysis, HAC / cluster covariances |
| **Distributions and graphics** | `distributions`, `viz`, `graphics`, `summary` | 20+ distributions, GEV, GPD, SVG plots, QQ / ACF / PACF, summary tables |

## The prelude

`use solow::prelude::*;` brings the everyday surface into scope:

- Error and numeric types from `solow_core`
- Formula-driven fit helpers (`ols`, `wls`, `gls`, `glm`, `logit`, `probit`, `poisson`)
- The workhorse estimators: `LinearModel`, `Glm`, `Logit`, `Probit`, `Poisson`, `LinearSvc`, `Svc`, `Svr`, `RandomForestClassifier`, `GradientBoostingRegressor`, `MlpClassifier`, `KMeans`, `Dbscan`, `Pca`, `Sarimax`, `Var`, and dozens more
- The everyday metrics: `mean_squared_error`, `r2_score`, `accuracy_score`, `roc_auc_score`, `log_loss`, plus `classification_report`, `silhouette_score`, `pairwise_distances`
- The cross-validation splitters and helpers: `KFold`, `StratifiedKFold`, `TimeSeriesSplit`, `PurgedKFold`, `cross_val_score`, `bootstrap_ci`

## Beyond the classical stack

Capabilities that few libraries expose as first-class modules.

- **Change-point detection.** `pelt` (Killick, Fearnhead, Eckley 2012), `cusum`, `binary_segmentation`.
- **Volatility.** `Garch11` with iterated multi-step variance forecast.
- **Extreme value analysis.** `Gev`, `Gpd` with `return_level(T)` and peaks-over-threshold fit.
- **Effect sizes.** `cohens_d`, `hedges_g`, `glass_delta`, `eta_squared`, `omega_squared`, `cliffs_delta`, `cramers_v`.
- **Meta-analysis.** `meta_fixed_effect`, `meta_random_effects` with Cochran Q, I², τ².
- **Control charts.** `ewma` and two-sided `cusum` with signed alarm streams.
- **Block bootstrap.** Moving, circular, stationary variants for time series.
- **Conformal prediction.** `SplitConformal`, `JackknifePlus` for distribution-free prediction intervals.

## Correctness

Every deterministic estimator is cross-verified against committed golden reference fixtures on every CI run.

- Closed-form solvers match bit-wise to `1e-10`.
- Iterative solvers match parameters to `1e-6` or predictions to `5e-2` where reference solvers themselves disagree at that scale.
- NIST StRD certified cases re-run on every CI. Worst-case certified relative error across the suite is `2.5e-10`. Longley (cond ~10¹⁰) matches certified coefficients to `~1e-13` because the QR/SVD path never forms XᵀX.
- `#![forbid(unsafe_code)]` on every crate.
- Every stochastic estimator is deterministic under a caller seed.

Run the full CI locally with `cargo test --workspace` (1000+ tests).

## Lean builds

The workspace ships 54 focused crates. Pick only what you need.

```toml
[dependencies]
solow-regression = "0.7"
solow-metrics    = "0.7"
solow-cv         = "0.7"
```

See the [full workspace list](https://github.com/benovamurat/solow/blob/main/docs/book/src/crates.md) and the [documentation site](https://benovamurat.github.io/solow/) for per-module walkthroughs.

## License

BSD-3-Clause. Copyright (c) 2026, Murat Ova ([Stochastic Minds](https://stochasticminds.com)).
