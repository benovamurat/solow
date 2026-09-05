# Statistical tests

The [`solow-stats`] crate is Solow's battery of hypothesis tests and regression
diagnostics. Every public quantity is cross-validated against the reference to
tight tolerances. This chapter shows the most common ones; the crate covers many
more (see the table at the end).

The examples reuse a fitted OLS model so the residuals are realistic:

```rust
use ndarray::{array, Array1};
use solow_core::tools::{add_constant, HasConstant};
use solow_regression::LinearModel;
use solow_stats::{durbin_watson, het_breuschpagan, jarque_bera};

let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
let y: Array1<f64> = array![1.0, 1.8, 3.3, 3.9, 5.2, 5.8, 7.4, 7.9];
let design = add_constant(&x, true, HasConstant::Add).unwrap();
let res = LinearModel::ols(y, design.clone()).unwrap().fit().unwrap();

// Autocorrelation of residuals (≈ 2 under no first-order autocorrelation).
let dw = durbin_watson(&res.resid);
println!("Durbin-Watson = {:.4}", dw);

// Normality of residuals.
let jb = jarque_bera(&res.resid);
println!("Jarque-Bera = {:.4} (p = {:.4})", jb.statistic, jb.pvalue);
println!("skew = {:.3}, kurtosis = {:.3}", jb.skew, jb.kurtosis);

// Heteroskedasticity (Breusch-Pagan): regress squared residuals on the design.
let (lm, lm_p, fval, f_p) = het_breuschpagan(&res.resid, &design).unwrap();
println!("Breusch-Pagan LM = {:.4} (p = {:.4})", lm, lm_p);
```

## Normality

- `jarque_bera(resid)` — the Jarque–Bera test, returning the statistic, its
  chi-squared(2) p-value, and the sample skewness and kurtosis.
- `omni_normtest(resid)` — D'Agostino–Pearson omnibus test.
- `kstest_normal` / `lilliefors` — Kolmogorov–Smirnov-style normality tests.

## Heteroskedasticity

- `het_breuschpagan(resid, exog_het)` — the Breusch–Pagan LM test. Returns
  `(lm, lm_pvalue, fvalue, f_pvalue)`.
- `het_white(resid, exog)` — White's test (squares and cross-products of the
  regressors); `exog` must contain a constant.
- `het_arch(resid, nlags)` — Engle's ARCH test for conditional
  heteroskedasticity.

## Autocorrelation

- `durbin_watson(resid)` — the Durbin–Watson statistic (between 0 and 4).
- `acorr_ljungbox(x, lags)` — the Ljung–Box test, one row per lag:

```rust
use ndarray::Array1;
use solow_stats::acorr_ljungbox;

let resid = Array1::from_vec(vec![
    0.1, -0.2, 0.05, 0.15, -0.1, 0.2, -0.05, 0.1, -0.15, 0.08,
]);
for row in acorr_ljungbox(&resid, 3) {
    println!("lag {}: LB = {:.4}, p = {:.4}", row.lag, row.lb_stat, row.lb_pvalue);
}
```

- `acorr_breusch_godfrey` and `acorr_lm` — Breusch–Godfrey / LM autocorrelation
  tests.

## Multiple-testing correction

`multipletests(pvals, alpha, method)` adjusts a family of p-values, returning
rejection flags and adjusted p-values in the original order:

```rust
use solow_stats::{multipletests, MultiTestMethod};

let pvals = [0.01, 0.04, 0.03, 0.2];
let adj = multipletests(&pvals, 0.05, MultiTestMethod::Holm);
println!("reject     = {:?}", adj.reject);
println!("adjusted p = {:?}", adj.pvals_corrected);
```

`MultiTestMethod` covers `Bonferroni`, `Holm`, and `FdrBh` (Benjamini–Hochberg).

## Two-sample and proportion tests

- `f_oneway` / `anova_oneway` — one-way ANOVA.
- `ttost_ind` — two one-sided tests (TOST) for equivalence.
- `proportions_ztest`, `proportion_confint` — proportion inference.
- `test_poisson_2indep` — comparison of two Poisson rates.

## Broader coverage

`solow-stats` also includes ANOVA tables (`anova_lm`), Tukey HSD (`tukey`),
the RESET specification test (`linear_reset`), variance inflation factors
(`variance_inflation_factor`), Cohen's and Fleiss' kappa, mediation analysis,
distance correlation, descriptive statistics (`describe`), and the robust
sandwich covariances (`cov_hc0`–`cov_hc3`, `cov_hac`, `cov_cluster`).

[`solow-stats`]: https://github.com/solow-rs/solow
