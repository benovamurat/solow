# Crate reference

Solow is a Cargo workspace of focused crates. Depend on the ones you need, or
pull in the umbrella `solow` crate to re-export the full public API. This page
lists the workspace.

## Foundation

| Crate | Purpose |
| --- | --- |
| `solow-core` | Error types, numeric aliases, shared data-handling tools (`add_constant`) |
| `solow-linalg` | Pure-Rust linear algebra (SVD, eigh, Cholesky, QR, LU, pinv) |
| `solow-distributions` | Special functions and statistical distributions, ECDF |
| `solow-optimize` | Newton / BFGS optimizers and numerical differentiation |

## Models

| Crate | Models |
| --- | --- |
| `solow-regression` | OLS, WLS, GLS, quantile, rolling/recursive LS, GLSAR, sliced inverse regression, robust covariances |
| `solow-glm` | Generalized linear models (families, links, IRLS) and Tweedie |
| `solow-discrete` | Logit, Probit, Poisson, MNLogit, negative-binomial, ordered, zero-inflated, conditional, truncated/hurdle |
| `solow-robust` | Robust linear models (M-estimation) |
| `solow-nonparametric` | Nonparametric smoothers (lowess, KDE, kernel regression) |
| `solow-multivariate` | PCA, factor analysis and rotation, MANOVA, canonical correlation |
| `solow-duration` | Survival analysis (Kaplan-Meier, Cox PH, log-rank) |
| `solow-mixed` | Linear mixed-effects models (REML) |
| `solow-gee` | Generalized estimating equations (incl. nominal/ordinal) |
| `solow-gam` | Generalized additive models (penalized splines) |
| `solow-impute` | Multiple imputation (Rubin's rules) |
| `solow-regime` | Regime-switching models (Markov switching) |
| `solow-othermod` | Beta regression |
| `solow-copula` | Copulas (Archimedean and elliptical) |
| `solow-bayes` | Bayesian mixed GLM (variational Bayes) |
| `solow-emplike` | Empirical-likelihood inference |

## Time series

| Crate | Contents |
| --- | --- |
| `solow-tsa` | acf/pacf, ADF, KPSS, AutoReg, STL, Holt-Winters, Granger, cointegration, HP/BK/CF filters, ARMA order selection |
| `solow-statespace` | Kalman filter/smoother, SARIMAX, unobserved components, dynamic factor |
| `solow-var` | VAR, SVAR, VECM / Johansen |

## Statistics, formulas, and presentation

| Crate | Purpose |
| --- | --- |
| `solow-stats` | Statistical tests and diagnostics (ANOVA, Tukey HSD, ARCH, RESET, robust sandwich covariances, VIF, mediation, and more) |
| `solow-formula` | R/patsy-style formula interface (design matrices) |
| `solow-summary` | Labeled results and summary tables |
| `solow-graphics` | Statistical graphics (qqplot, plot_acf, influence) |
| `solow-viz` | General-purpose data-visualization backend (SVG) |

## Umbrella and tooling

| Crate | Purpose |
| --- | --- |
| `solow` | Umbrella crate re-exporting the full public API, with a `prelude` |
| `solow-py` | PyO3 bindings: import `solow` from Python |
| `solow-bench` | `criterion` benchmark harness |

Every library crate is cross-validated against an authoritative reference
implementation — most quantities to `1e-8`, MLE/VB estimates to `1e-6`, the
formula engine to `1e-12`, and the Python bindings to about `1e-12`.
