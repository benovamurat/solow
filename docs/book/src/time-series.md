# Time series

Solow's time-series functionality spans three crates:

- [`solow-tsa`] — sample statistics (acf/pacf), unit-root tests, and the
  autoregressive estimator `AutoReg`.
- [`solow-statespace`] — the Kalman filter/smoother and the SARIMAX seasonal
  ARIMA estimator.
- [`solow-var`] — vector autoregression (VAR), SVAR, and VECM/Johansen.

## Autocorrelation: acf and pacf

`acf` returns the autocorrelation function out to `nlags` (lag 0 is `1.0`);
`pacf` returns the partial autocorrelation function via a chosen
[`PacfMethod`].

```rust
use ndarray::Array1;
use solow_tsa::{acf, pacf, PacfMethod};

let y = Array1::from_vec(vec![
    0.0, 0.4, 0.1, 0.5, 0.2, 0.6, 0.3, 0.7, 0.35, 0.75, 0.4, 0.8,
]);

// Autocorrelations for lags 0..=4 (adjusted = false → divide by n).
let a = acf(&y, 4, false).unwrap();
assert!((a[0] - 1.0).abs() < 1e-12);
println!("acf = {:?}", a);

// Partial autocorrelations via Yule–Walker.
let p = pacf(&y, 4, PacfMethod::YuleWalker).unwrap();
println!("pacf = {:?}", p);
```

`PacfMethod::YuleWalker` uses the sample-size-adjusted autocovariance;
`PacfMethod::Ols` regresses the series on its lags plus a constant.

## Unit-root tests

The augmented Dickey–Fuller test is `adfuller`; KPSS is `kpss`.

```rust
use ndarray::Array1;
use solow_tsa::{adfuller, AdfRegression, AutoLag};

let y = Array1::from_vec(vec![
    1.0, 1.2, 0.9, 1.4, 1.1, 1.6, 1.3, 1.8, 1.5, 2.0, 1.7, 2.2, 1.9, 2.4,
]);

let res = adfuller(
    &y,
    1,                         // maxlag (upper bound on lagged differences)
    AdfRegression::C,          // constant only
    AutoLag::Aic,              // choose lag length by AIC
)
.unwrap();
println!("ADF statistic = {:.4}, p-value = {:.4}", res.adfstat, res.pvalue);
println!("lags used = {}, nobs = {}", res.usedlag, res.nobs);
```

## Autoregression: AutoReg

`AutoReg::new(endog, lags, trend)` builds an AR(`lags`) model with a
deterministic [`Trend`] term (`Trend::N`, `Trend::C`, `Trend::T`, `Trend::Ct`,
`Trend::Ctt`). `.fit()` estimates it.

```rust
use ndarray::Array1;
use solow_tsa::{AutoReg, Trend};

let y = Array1::from_vec(vec![
    0.2, 0.5, 0.1, 0.6, 0.3, 0.7, 0.35, 0.75, 0.4, 0.8, 0.45, 0.85,
]);

let res = AutoReg::new(y, 1, Trend::C).unwrap().fit().unwrap();
assert_eq!(res.params.len(), 2); // const + 1 lag
println!("AR(1) params = {:?}", res.params);
println!("AIC = {:.2}", res.aic);
```

To pick the lag order automatically, `ar_select_order` scans candidate orders by
an information criterion ([`ArIc`]):

```rust
use ndarray::Array1;
use solow_tsa::{ar_select_order, ArIc, Trend};

let y = Array1::from_vec(vec![
    0.0, 0.4, 0.1, 0.5, 0.2, 0.6, 0.3, 0.7, 0.35, 0.75, 0.4, 0.8, 0.45, 0.85,
]);

let sel = ar_select_order(&y, 3, ArIc::Aic, Trend::C).unwrap();
println!("selected order = {}", sel.selected_order);
println!("IC path = {:?}", sel.ic_path);
```

## Seasonal ARIMA: SARIMAX

[`Sarimax`] estimates a seasonal ARIMA model by maximum likelihood on top of the
Kalman filter. The order is given by a [`SarimaxOrder`]: `SarimaxOrder::new(p,
d, q)` for a non-seasonal model, or `SarimaxOrder::seasonal(p, d, q, sp, sd, sq,
s)` for the full `(p, d, q)(P, D, Q, s)` specification.

```rust
use ndarray::Array1;
use solow_statespace::{Sarimax, SarimaxOrder};

let y = Array1::from_vec(vec![
    0.2, 0.5, 0.1, -0.3, 0.4, 0.8, 0.3, -0.1, 0.0, 0.6, 0.9, 0.2, -0.4, 0.1,
]);

// ARIMA(1, 0, 0).
let res = Sarimax::new(y.clone(), SarimaxOrder::new(1, 0, 0))
    .unwrap()
    .fit()
    .unwrap();
println!("params = {:?}", res.params); // [ar.L1, sigma2]
println!("llf = {:.4}, aic = {:.2}", res.llf, res.aic);
```

A seasonal example, `(1, 1, 1)(1, 0, 0, 12)`:

```rust
use ndarray::Array1;
use solow_statespace::{Sarimax, SarimaxOrder};

# let y = Array1::from_vec((0..48).map(|i| (i as f64 * 0.5).sin()).collect());
let order = SarimaxOrder::seasonal(1, 1, 1, 1, 0, 0, 12);
let res = Sarimax::new(y, order).unwrap().fit().unwrap();
println!("converged llf = {:.4}", res.llf);
```

## Vector autoregression: VAR

[`Var::new(endog)`] takes a `T × K` matrix (rows = time, columns = variables);
`.fit(p)` estimates a VAR(`p`) by equation-by-equation OLS.

```rust
use ndarray::array;
use solow_var::Var;

// Two series over twelve periods.
let y = array![
    [0.5, 1.0], [0.7, 0.8], [0.4, 1.2], [0.9, 0.6], [0.6, 1.1],
    [1.0, 0.5], [0.7, 0.9], [1.1, 0.4], [0.8, 1.0], [1.2, 0.3],
    [0.9, 0.8], [1.3, 0.5],
];

let res = Var::new(y).unwrap().fit(1).unwrap();
assert_eq!(res.neqs, 2);
assert_eq!(res.coefs.len(), 1); // one lag matrix
println!("AIC = {:.4}, BIC = {:.4}", res.aic, res.bic);
```

`VarResults` exposes the per-lag coefficient matrices (`coefs`), both the ML and
degrees-of-freedom-adjusted residual covariances, the log-likelihood, the
AIC/BIC/HQIC/FPE information criteria, and per-coefficient standard errors,
t-statistics, and p-values.

For structural VARs use [`Svar`]; for cointegrated systems use [`Vecm`] together
with the Johansen test [`coint_johansen`].

## Granger causality

```rust
use ndarray::array;
use solow_tsa::grangercausalitytests;

// Column 0 is the "caused" series, column 1 the candidate cause.
let data = array![
    [0.5, 1.0], [0.7, 0.8], [0.4, 1.2], [0.9, 0.6], [0.6, 1.1],
    [1.0, 0.5], [0.7, 0.9], [1.1, 0.4], [0.8, 1.0], [1.2, 0.3],
    [0.9, 0.8], [1.3, 0.5],
];

let results = grangercausalitytests(&data, 2).unwrap();
for r in &results {
    // ssr_ftest is (F statistic, p-value, df_num, df_denom).
    let (f, p, _df_num, _df_den) = r.ssr_ftest;
    println!("lag {}: F = {:.3}, p = {:.3}", r.lag, f, p);
}
```

[`solow-tsa`]: https://github.com/solow-rs/solow
[`solow-statespace`]: https://github.com/solow-rs/solow
[`solow-var`]: https://github.com/solow-rs/solow
[`Trend`]: ./time-series.md
[`ArIc`]: ./time-series.md
[`PacfMethod`]: ./time-series.md
[`Sarimax`]: ./time-series.md
[`SarimaxOrder`]: ./time-series.md
[`Var::new(endog)`]: ./time-series.md
[`Svar`]: ./time-series.md
[`Vecm`]: ./time-series.md
[`coint_johansen`]: ./time-series.md
