<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/benovamurat/solow/main/assets/solow-logo-dark.svg">
    <img alt="Solow" src="https://raw.githubusercontent.com/benovamurat/solow/main/assets/solow-logo.svg" width="320">
  </picture>
</p>

<p align="center">
  <strong>A complete statistical-modeling, econometrics &amp; data-visualization toolkit for Rust.</strong><br>
  Faithful, fully-verified, from-scratch — validated to near machine precision.
</p>

<p align="center">
  <a href="https://github.com/benovamurat/solow/actions/workflows/ci.yml"><img src="https://github.com/benovamurat/solow/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://benovamurat.github.io/solow/"><img src="https://img.shields.io/badge/docs-mdBook-success.svg" alt="Docs"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-BSD--3--Clause-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/rust-1.80%2B-dea584.svg" alt="Rust 1.80+">
  <img src="https://img.shields.io/badge/unsafe-forbidden-success.svg" alt="unsafe forbidden">
</p>

<p align="center">
  <a href="https://benovamurat.github.io/solow/"><b>Documentation</b></a> ·
  <a href="https://benovamurat.github.io/solow/examples/index.html"><b>Examples</b></a> ·
  <a href="docs/VALIDATION.md"><b>Validation</b></a> ·
  <a href="docs/BENCHMARKS.md"><b>Benchmarks</b></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/benovamurat/solow/main/docs/book/src/examples/img/ols.svg" width="270">
  <img src="https://raw.githubusercontent.com/benovamurat/solow/main/docs/book/src/examples/img/case_forecasting.svg" width="270">
  <img src="https://raw.githubusercontent.com/benovamurat/solow/main/docs/book/src/examples/img/state_space.svg" width="270">
</p>
<p align="center"><sub>Regression · forecasting with prediction bands · the Kalman filter — every figure is rendered by <code>solow-viz</code>, the built-in dependency-light SVG backend.</sub></p>

---

## Why Solow

- **Comprehensive.** 30+ focused crates covering linear & generalized linear models, discrete choice, robust regression, time series & state space, survival, multivariate, mixed effects, GEE, GAM, copulas, Bayesian VB, and a full battery of statistical tests — the surface of a mature statistics library.
- **Correct, and it proves it.** Every estimator is cross-validated against an authoritative reference to **~1e-8 or tighter**, plus NIST StRD certified cases. The golden fixtures are committed, so every CI run re-verifies against the exact same ground truth.
- **Self-contained.** The numerical core — SVD, eigendecomposition, Cholesky/QR/LU, special functions, and the distributions — is pure Rust with **no system LAPACK/BLAS**. The only foundational dependency is `ndarray`. `#![forbid(unsafe_code)]`.
- **Familiar.** Results print the canonical `.summary()` table, so output reads exactly like the reference you already know.
- **Production-ready.** Memory-safe statistical inference in a single binary, with **no Python runtime** — fit a risk model or a forecasting service entirely in Rust.
- **Bridges your stack.** Fit a model straight from a **Polars** `DataFrame`, or call Solow from **Python** through the PyO3 bindings.

## A taste

```rust
use solow_core::tools::{add_constant, HasConstant};
use solow_regression::LinearModel;

// `x`, `y` are ndarray columns — here, 50 noisy points of y ≈ 2 + 0.5·x.
let design = add_constant(&x, true, HasConstant::Add)?;
let res = LinearModel::ols(y, design)?.fit()?;

println!("{}", res.summary(Some(&["const", "x"])));
```

```text
                            OLS Regression Results
==============================================================================
Dep. Variable:                       y   R-squared:                     0.976
Model:                             OLS   Adj. R-squared:                0.975
Method:                  Least Squares   F-statistic:                    1924
Date:                 Thu, 18 Jun 2026   Prob (F-statistic):         2.17e-40
Time:                         13:16:20   Log-Likelihood:              -77.253
No. Observations:                   50   AIC:                           158.5
Df Residuals:                       48   BIC:                           162.3
Df Model:                            1
Covariance Type:             nonrobust
==============================================================================
                   coef    std err         t     P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const            2.1421      0.323     6.640     0.000       1.493       2.791
x                0.4977      0.011    43.864     0.000       0.475       0.521
==============================================================================
Omnibus:                         1.459   Durbin-Watson:                 1.712
Prob(Omnibus):                   0.482   Jarque-Bera (JB):              1.079
Skew:                            0.048   Prob(JB):                      0.583
Kurtosis:                        2.287   Cond. No.                       56.1
==============================================================================
```

The [examples gallery](https://benovamurat.github.io/solow/examples/index.html) has 20+ runnable end-to-end vignettes — each with example data, the fitted summary, and a plot.

## Get your data in

**From Polars.** `solow-polars` fits a model in one call straight off a `DataFrame`:

```rust
use solow_polars::ols_from_frame;
let res = ols_from_frame(&df, "sales", &["ad_spend", "price"], true)?;
println!("R² = {:.3}", res.rsquared);
```

**From Python.** `solow-py` exposes the library through PyO3 — `pip install maturin && maturin develop`, then `import solow` and fit models with the same numerics, verified to reproduce reference outputs to ~1e-12.

## Correctness

Correctness is the whole point of Solow, so it is measured, committed, and reproducible rather than asserted:

- **Golden fixtures, near machine precision.** Most quantities agree to **~1e-8** or tighter; MLE / variational estimates to ~1e-6.
- **NIST StRD certified cases.** Worst-case certified relative error across the suite is **2.5e-10**; the ill-conditioned Longley design (cond ~10¹⁰) is matched to **~1e-13**, because the QR/SVD path never forms `XᵀX`.
- **Canonical real datasets.** Longley, Brownlee stack-loss, Spector & Mazzeo, capital-punishment counts, and Scottish devolution reproduce the reference to **~1e-11** on coefficients.
- **Honest about the floor.** [`docs/VALIDATION.md`](docs/VALIDATION.md) names the few quantities that do *not* reach machine precision and attributes each to the conditioning of that design. Reproduce with `cargo test -p solow --test validation -- --nocapture` — the NIST half depends on nothing but NIST.

## What's inside

<details>
<summary><b>The workspace — 30+ crates</b> (click to expand)</summary>

| Crate                  | Purpose                                                        |
| ---------------------- | ------------------------------------------------------------- |
| `solow-core`           | Error types, numeric aliases, shared data-handling tools      |
| `solow-linalg`         | Pure-Rust linear algebra (SVD, eigh, Cholesky, QR, LU, pinv)  |
| `solow-distributions`  | Special functions and statistical distributions               |
| `solow-optimize`       | Newton / BFGS optimizers and numerical differentiation        |
| `solow-regression`     | Linear regression (OLS, WLS, GLS, GLSAR, quantile, rolling)   |
| `solow-glm`            | Generalized linear models (families, links, IRLS, Tweedie)    |
| `solow-discrete`       | Discrete choice & counts (Logit, Probit, Poisson, MNLogit, NB, ordered, zero-inflated) |
| `solow-stats`          | Statistical tests, diagnostics, robust (HC/HAC) covariances   |
| `solow-tsa`            | Time series (acf/pacf, ADF, AutoReg, ARMA, STL, filters)      |
| `solow-robust`         | Robust linear models (M-estimation)                           |
| `solow-nonparametric`  | Nonparametric smoothers (lowess, KDE, kernel regression)      |
| `solow-multivariate`   | PCA, factor analysis, MANOVA, canonical correlation           |
| `solow-duration`       | Survival analysis (Kaplan-Meier, Cox PH, log-rank)            |
| `solow-statespace`     | Kalman filter/smoother, SARIMAX, unobserved components        |
| `solow-var`            | Vector autoregression (VAR, VECM/Johansen, SVAR)              |
| `solow-mixed`          | Linear mixed-effects models (REML)                            |
| `solow-gee`            | Generalized estimating equations (incl. nominal/ordinal)      |
| `solow-gam`            | Generalized additive models (penalized splines)               |
| `solow-impute`         | Multiple imputation (Rubin's rules)                           |
| `solow-graphics`       | Statistical graphics (qqplot, plot_acf, influence)            |
| `solow-regime`         | Regime-switching models (Markov switching)                    |
| `solow-othermod`       | Beta regression                                               |
| `solow-copula`         | Copulas (Archimedean and elliptical)                          |
| `solow-bayes`          | Bayesian mixed GLM (variational Bayes)                        |
| `solow-emplike`        | Empirical-likelihood inference                                |
| `solow-formula`        | R/patsy-style formula interface (design matrices)             |
| `solow-summary`        | Labeled results and summary tables                            |
| `solow-viz`            | General-purpose data-visualization backend (SVG)              |
| `solow`                | Umbrella crate re-exporting the full public API               |
| `solow-py`             | PyO3 bindings (import from Python)                             |
| `solow-polars`         | Polars `DataFrame` interop                                     |
| `solow-bench`          | `criterion` benchmark harness                                 |

</details>

The full library compiles with the test suite passing, every model cross-validated against an authoritative reference (golden values independently confirmed genuine). The formula engine is validated against `patsy` to 1e-12.

## Status

`0.1.x`, under active development. See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the plan and the state of every module.

## About

Solow is designed and built by **Murat Ova** at **Stochastic Minds**. Design notes and essays live at **Product Philosophy**.

- Company: Stochastic Minds — https://stochasticminds.com
- Writing: Product Philosophy — https://productphilosophy.com

## License

BSD-3-Clause. Copyright (c) 2026, Murat Ova (Stochastic Minds). See [`LICENSE`](LICENSE).
