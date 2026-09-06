# solow-statespace

**State-space models for Rust.** Kalman filter and smoother, SARIMAX, unobserved-components structural time series, dynamic factor models, multivariate state space.

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-statespace = "0.7"
```

## Quick example

```rust
use solow_statespace::Sarimax;

let m = Sarimax::builder(y)
    .order(1, 1, 1)
    .seasonal(1, 1, 1, 12)
    .exog(x)
    .fit()?;

println!("{}", m.summary());
let fc = m.forecast(24)?;
```

## What is inside

| Model | What it does |
|---|---|
| `StateSpace` | Generic time-invariant linear state-space model with univariate observations. Kalman filter + fixed-interval smoother. |
| `MvStateSpace` | Multivariate observation variant of the above. |
| `Sarimax` | Seasonal ARIMA with exogenous regressors, fit by exact maximum likelihood through the Kalman likelihood. |
| `UnobservedComponents` | Structural time series (local level / local linear trend / seasonal / cycle) with configurable component set. |
| `DynamicFactor` | Dynamic factor model for panels of observed series with a small number of latent factors. |

### Every fit returns

- Kalman-filtered and smoothed states with per-step covariance.
- One-step-ahead prediction errors and their variances.
- Log-likelihood, AIC, BIC.
- Multi-step-ahead forecasts with prediction intervals.

## Correctness

- Cross-verified against committed golden reference fixtures on every CI run.
- Ill-conditioned SARIMAX seasonalities produce numerically stable filter passes via the square-root Kalman path.
- `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.
