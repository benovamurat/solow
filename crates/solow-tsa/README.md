# solow-tsa

**Time series analysis for Rust.** ACF, PACF, ARMA, AutoReg, STL, Holt-Winters, HP / BK / CF filters, unit-root and cointegration tests, GARCH(1,1) volatility, change-point detection (CUSUM / PELT / Binary Segmentation), EWMA control chart.

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-tsa = "0.7"
```

## Quick example

```rust
use solow_tsa::{AutoReg, adfuller, pelt};

let ar = AutoReg::new(&y).max_lags(12).ic("aic").fit()?;
let (adf_stat, pval, _) = adfuller(&y, None)?;
let change_points = pelt(&y, 3.0)?;
```

## What is inside

### Autocorrelation and lag design

`acf`, `pacf`, `ccf`, Ljung-Box Q, `lagmat`, `add_trend`.

### Unit root and cointegration

| Function | What it does |
|---|---|
| `adfuller` | Augmented Dickey-Fuller unit root test with lag selection. |
| `kpss` | Kwiatkowski-Phillips-Schmidt-Shin stationarity test. |
| `coint` | Engle-Granger two-step cointegration test. |
| `zivot_andrews` | Unit root test allowing a single structural break. |
| `range_unit_root` | Range unit root test. |
| `granger_causality` | Wald test for Granger causality. |

### AR, ARMA, and smoothing

| Estimator | What it does |
|---|---|
| `AutoReg` | Autoregression AR(p) with automatic order selection (AIC or BIC). |
| `ArmaProcess` | ARMA(p, q) process arithmetic and simulation. |
| `SimpleExpSmoothing`, `Holt`, `ExponentialSmoothing` | Exponential smoothing and Holt-Winters. |

### Decomposition and filters

| Function | What it does |
|---|---|
| `STL` | Seasonal-trend decomposition using LOESS, robust variant supported. |
| `seasonal_decompose` | Classical additive or multiplicative decomposition. |
| `hp_filter` | Hodrick-Prescott filter. |
| `bk_filter` | Baxter-King band-pass filter. |
| `cf_filter` | Christiano-Fitzgerald asymmetric filter. |

### Volatility

| Estimator | What it does |
|---|---|
| `Garch11` | GARCH(1, 1) MLE with iterated multi-step variance forecast. |

### Change-point detection

| Function | What it does |
|---|---|
| `pelt` | Killick-Fearnhead-Eckley 2012 PELT (pruned exact linear time). |
| `cusum` | CUSUM detector for a mean shift. |
| `binary_segmentation` | Recursive binary segmentation. |

### Control charts

| Function | What it does |
|---|---|
| `ewma` | Roberts 1959 EWMA control chart with signed alarm stream. |
| Two-sided `cusum` | Bidirectional CUSUM with configurable reference and decision interval. |

### Heteroskedasticity

`breakvar_heteroskedasticity`, `arch_lm_test`.

## Correctness

- Every estimator cross-verified against committed golden reference fixtures.
- `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.
