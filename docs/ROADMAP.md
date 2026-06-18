# Solow — Roadmap
A faithful, fully-verified, from-scratch Rust re-implementation of the canonical
statistical-computing stack. **733** public symbols are catalogued in
[`API_INVENTORY.md`](API_INVENTORY.md); this file tracks the build order.

## Verification contract
Every model emits the same quantities as the reference and is checked against
golden JSON fixtures generated from an authoritative reference implementation
(`tools/reference/`). Tolerances target ~1e-10. No claim of "done" without a
passing fixture test.

## Scale

| Priority | Count |
|---|---|
| P0 (foundational) | 169 |
| P1 | 204 |
| P2 | 198 |
| P3 (niche) | 162 |
| **Total** | **733** |

## Phases

### Phase 0 — Numeric foundation — _DONE_

`solow-core`, `solow-linalg` (SVD/eigh/Cholesky/QR/LU/pinv), `solow-distributions` (normal/t/F/chi2 + special functions). All verified.

| Area | P0 | P1 | P2 | P3 |
|---|---|---|---|---|
| distributions (core) | 7 | 10 | 8 | 7 |

### Phase 1 — Linear regression & inference — _DONE (OLS/WLS/GLS verified)_

`solow-regression`: OLS/WLS/GLS + the full results/inference layer (cov_params, bse,
t-tests, conf_int, R²/adj, F-test, AIC/BIC, sums of squares, summary table) — verified
to 1e-7..1e-9. `solow-viz` core (figure/axes/line/scatter/bar/hist + SVG) done.
Remaining for later: GLSAR, robust/HC covariance types, F-/Wald-test helpers.

| Area | P0 | P1 | P2 | P3 |
|---|---|---|---|---|
| regression-linear | 4 | 4 | 5 | 2 |
| base | 9 | 9 | 12 | 10 |
| tools | 11 | 13 | 18 | 29 |

### Phase 2 — GLM, discrete choice, core stats — _DONE (core verified)_

`solow-glm` **DONE & verified**: exponential families (Gaussian, Binomial, Poisson,
Gamma, Inverse-Gaussian, Negative-Binomial), 8 link functions, IRLS estimator, full
results (deviance, Pearson χ², null deviance, llf, AIC/BIC, scale, all residual types,
conf_int). Gaussian/Poisson/Binomial-logit/Binomial-probit/Gamma-log all match the
reference at the MLE.
**DONE & verified:** `solow-optimize` (Newton/BFGS + numerical differentiation),
`solow-discrete` (Logit/Probit/Poisson via Newton, with llnull/llr/pseudo-R²),
`solow-stats` (Jarque-Bera, Durbin-Watson, omnibus normality, Breusch-Pagan, White,
Ljung-Box, weighted stats, t/z tests, multiple-testing corrections).
Remaining: MNLogit & negative-binomial discrete models, more stats tests (power, anova,
contingency, multicomp).

| Area | P0 | P1 | P2 | P3 |
|---|---|---|---|---|
| genmod-glm | 14 | 5 | 8 | 8 |
| discrete | 8 | 8 | 12 | 15 |
| stats-core | 11 | 10 | 7 | 12 |
| regression-other | 4 | 5 | 7 | 6 |

### Phase 3 — Time series & robust models — _DONE (core verified)_

**DONE & verified:** `solow-tsa` (acovf/acf/pacf/ccf, Ljung-Box Q, ADF, lagmat/add_trend,
AutoReg), `solow-robust` (RLM with Huber/Tukey/Andrew M-estimators, MAD/Huber scale).
Remaining: KPSS, ARIMA/SARIMAX, VAR/VECM, Holt-Winters/ETS, regime switching.

| Area | P0 | P1 | P2 | P3 |
|---|---|---|---|---|
| tsa-stattools | 10 | 18 | 12 | 11 |
| tsa-arima | 8 | 12 | 8 | 3 |
| tsa-var | 14 | 13 | 9 | 4 |
| robust | 5 | 5 | 6 | 5 |

### Phase 4 — State space, nonparametric, multivariate, duration — _DONE (core verified)_

**DONE & verified:** `solow-nonparametric` (lowess, KDE, bandwidth selectors),
`solow-multivariate` (PCA, Factor analysis, MANOVA, CanCorr), `solow-duration`
(Kaplan-Meier, Cox PH Breslow), `solow-statespace` (Kalman filter/smoother + SARIMAX
by ML), and the full `solow-stats` test battery (anova_lm, oneway/Welch, proportions,
Tukey HSD, power, contingency).
Remaining (long tail): dynamic factor / structural state-space variants, kernel
regression (`KernelReg`), MANOVA contrasts.

| Area | P0 | P1 | P2 | P3 |
|---|---|---|---|---|
| tsa-statespace | 14 | 13 | 6 | 1 |
| nonparametric | 8 | 10 | 8 | 2 |
| multivariate | 1 | 7 | 5 | 1 |
| duration | 4 | 4 | 3 | 2 |
| stats-tests | 17 | 29 | 27 | 18 |

### Phase 5 — GEE, mixed, GAM, imputation, advanced graphics — _DONE (core verified)_

**DONE & verified:** `solow-gee` (exchangeable/independence working correlation, robust
sandwich SE), `solow-mixed` (linear mixed-effects via REML), `solow-gam` (GLMGam
penalized B-splines, canonical links), `solow-impute` (Rubin's combining rules +
deterministic imputation), `solow-graphics` (qqplot/ProbPlot, plot_acf/pacf), and the
`solow-var` (VAR) + `solow-tsa` Holt-Winters/ETS additions, `solow-discrete` MNLogit /
negative binomial.
Remaining (long tail): VECM/Johansen, SVAR, regime-switching, Bayesian mixed GLM,
nominal/ordinal GEE, non-canonical GAM links, MANOVA/mediation extras, mosaic plots.

| Area | P0 | P1 | P2 | P3 |
|---|---|---|---|---|
| genmod-gee | 5 | 1 | 10 | 9 |
| gam-imputation-other | 8 | 14 | 10 | 7 |
| graphics | 7 | 14 | 17 | 10 |
