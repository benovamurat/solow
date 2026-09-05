# Migration from the reference (Python)

If you are coming from the canonical Python statistics stack — NumPy/SciPy plus
the standard econometrics-modeling package that Solow re-implements — most of
your knowledge transfers directly. The model names, the result attributes, and
the formula syntax are intentionally familiar. This chapter maps the Python
symbols you know to their Solow equivalents and flags the handful of real
differences.

## Conventions that carry over

- **You supply the intercept.** Just like the reference's array API (and unlike
  its formula API), Solow does not add a constant column automatically. Use
  `solow_core::tools::add_constant`, or let the [formula interface](./formula.md)
  add `Intercept` for you.
- **Result attributes have the same names.** `params`, `bse`, `tvalues`,
  `pvalues`, `rsquared`, `aic`, `bic`, `llf`, `fittedvalues`, `resid`,
  `conf_int(...)` — all present, all meaning the same thing.
- **The formula DSL is patsy-compatible.** `y ~ x1 * x2 + C(g)` builds the same
  design matrix, column for column.

## The two real differences

1. **`model(endog, exog)` argument order**, matching the reference exactly:
   the response comes first, the design second. Solow keeps that order:
   `LinearModel::ols(y, x)`.
2. **Construct-then-`.fit()`** returns a `Result`, so you call `.unwrap()` (or
   propagate the error with `?`). There is no silent failure.

## Symbol map

The Python column shows the reference's public path; the Solow column shows the
Rust equivalent.

### Linear models

| Reference (Python) | Solow (Rust) |
| --- | --- |
| `OLS(y, X).fit()` | `LinearModel::ols(y, x)?.fit()?` |
| `WLS(y, X, weights=w).fit()` | `LinearModel::wls(y, x, w)?.fit()?` |
| `GLS(y, X, sigma=S).fit()` | `LinearModel::gls(y, x, &s)?.fit()?` |
| `res.get_robustcov_results('HC3')` | `res.cov_params_robust(&x, &CovType::Hc3)?` |
| `cov_type='HAC', maxlags=L` | `CovType::Hac { maxlags: L, use_correction }` |
| `cov_type='cluster', groups=g` | `CovType::Cluster { groups, use_correction }` |
| `QuantReg(y, X).fit(q=0.5)` | `QuantReg` (in `solow-regression`) |
| `RollingOLS`, `RecursiveLS` | `RollingOLS`, `RecursiveLS` |

### Generalized linear models

| Reference (Python) | Solow (Rust) |
| --- | --- |
| `GLM(y, X, family=Poisson()).fit()` | `Glm::new(y, x, Family::Poisson)?.fit()?` |
| `family=Binomial(link=probit())` | `Glm::with_link(y, x, Family::Binomial, Link::Probit)?` |
| `family=Gamma()` | `Family::Gamma` (default `Link::InversePower`) |
| `family=NegativeBinomial(alpha=a)` | `Family::NegativeBinomial { alpha: a }` |
| `res.deviance`, `res.pearson_chi2` | `res.deviance`, `res.pearson_chi2` |
| `res.pseudo_rsquared()` | `res.pseudo_rsquared()` |

### Discrete choice & counts

| Reference (Python) | Solow (Rust) |
| --- | --- |
| `Logit(y, X).fit()` | `Logit::new(y, x)?.fit()?` |
| `Probit(y, X).fit()` | `Probit::new(y, x)?.fit()?` |
| `Poisson(y, X).fit()` | `Poisson::new(y, x)?.fit()?` (in `solow-discrete`) |
| `MNLogit`, `NegativeBinomial` | `MNLogit`, `NegativeBinomial` |
| `res.prsquared`, `res.llr_pvalue` | `res.prsquared`, `res.llr_pvalue` |

### Robust regression

| Reference (Python) | Solow (Rust) |
| --- | --- |
| `RLM(y, X, M=norms.TukeyBiweight()).fit()` | `Rlm::new(y, x, TukeyBiweight::default())?.fit()?` |
| `M=norms.HuberT()` | `HuberT::default()` |

### Time series

| Reference (Python) | Solow (Rust) |
| --- | --- |
| `acf(x, nlags=k)` | `acf(&x, k, adjusted)?` |
| `pacf(x, nlags=k, method='yw')` | `pacf(&x, k, PacfMethod::YuleWalker)?` |
| `adfuller(x, ...)` | `adfuller(&x, maxlag, AdfRegression::C, AutoLag::Aic)?` |
| `kpss(x, ...)` | `kpss(...)` |
| `AutoReg(x, lags=p, trend='c').fit()` | `AutoReg::new(x, p, Trend::C)?.fit()?` |
| `ar_select_order(x, maxlag)` | `ar_select_order(&x, maxlag, ArIc::Aic, Trend::C)?` |
| `SARIMAX(y, order=(p,d,q)).fit()` | `Sarimax::new(y, SarimaxOrder::new(p, d, q))?.fit()?` |
| `SARIMAX(..., seasonal_order=(P,D,Q,s))` | `SarimaxOrder::seasonal(p, d, q, P, D, Q, s)` |
| `VAR(Y).fit(p)` | `Var::new(y)?.fit(p)?` |
| `grangercausalitytests(data, maxlag)` | `grangercausalitytests(&data, maxlag)?` |

### Survival

| Reference (Python) | Solow (Rust) |
| --- | --- |
| `SurvfuncRight(time, status)` | `SurvfuncRight::new(&time, &status)?` |
| `PHReg(time, exog, status).fit()` | `PHReg::new(&time, &exog, &status)?.fit()?` |
| `survdiff(time, status, group)` | `survdiff(&time, &status, &group, WeightType::LogRank)?` |

### Multivariate & nonparametric

| Reference (Python) | Solow (Rust) |
| --- | --- |
| `PCA(data)` | `Pca::new(data).fit()?` |
| `Factor(data, n_factor=k).fit()` | `Factor::from_data(&data, k, true).fit(maxiter, tol)?` |
| `lowess(y, x, frac=f)` | `lowess(&y, &x, LowessOptions { frac, .. })?` |
| `KDEUnivariate(x).fit()` | `KdeUnivariate::new(x).fit(Bandwidth::NormalReference)?` |

### Tests & distributions

| Reference (Python) | Solow (Rust) |
| --- | --- |
| `durbin_watson(resid)` | `durbin_watson(&resid)` |
| `jarque_bera(resid)` | `jarque_bera(&resid)` |
| `het_breuschpagan(resid, exog)` | `het_breuschpagan(&resid, &exog)?` |
| `acorr_ljungbox(x, lags=k)` | `acorr_ljungbox(&x, k)` |
| `multipletests(p, method='holm')` | `multipletests(&p, alpha, MultiTestMethod::Holm)` |
| `scipy.stats.norm.cdf(x)` | `norm_cdf(x)` or `Normal::new(0.0, 1.0).cdf(x)` |
| `scipy.stats.t.sf(x, df)` | `t_sf(x, df)` |
| `scipy.special.gammaln(x)` | `lgamma(x)` |

## A side-by-side example

Reference (Python):

```python
import numpy as np
# `sm` is the reference econometrics package
X = sm.add_constant(x)
res = sm.OLS(y, X).fit()
print(res.params, res.rsquared)
print(res.get_robustcov_results('HC3').bse)
```

Solow (Rust):

```rust
use ndarray::{array, Array1};
use solow_core::tools::{add_constant, HasConstant};
use solow_regression::{CovType, LinearModel};

let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
let y: Array1<f64> = array![1.1, 1.9, 3.2, 3.9, 5.1];

let design = add_constant(&x, true, HasConstant::Add).unwrap();
let res = LinearModel::ols(y, design.clone()).unwrap().fit().unwrap();
println!("{:?}  R^2={:.4}", res.params, res.rsquared);
println!("HC3 bse = {:?}", res.bse_robust(&design, &CovType::Hc3).unwrap());
```

## Using Solow from Python instead

If you would rather keep writing Python but get Solow's pure-Rust engine
underneath, the `solow-py` bindings expose `OLS`, `WLS`, `GLS`, `GLM`, `Logit`,
`Probit`, `Poisson`, `AutoReg`, `acf`, and `pacf` as a NumPy-friendly `solow`
module. See [Using Solow from Python](./python.md).
