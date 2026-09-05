# Robust regression

The [`solow-robust`] crate estimates linear models by **M-estimation** — the
robust linear model (RLM). Instead of minimizing the sum of squared residuals
(which a single gross outlier can dominate), it minimizes a robust criterion
`Σ ρ((yᵢ − xᵢ·β) / σ)` for a bounded-influence function `ρ`, re-estimating the
scale `σ` from the residuals at each step. The optimization is iteratively
reweighted least squares (IRLS), exactly mirroring the reference's RLM.

> Robust *covariances* for an ordinary least-squares fit (HC0–HC3, HAC,
> cluster) live in [`solow-regression`] and are covered in the
> [Linear regression](./regression.md#robust-sandwich-covariances) chapter.
> This chapter is about robust *point estimation*.

## Robust criteria (norms)

The norm decides how residuals are down-weighted:

| Norm | Behaviour | Constructor |
| --- | --- | --- |
| `norms::HuberT` | Monotone: large residuals get linear (not quadratic) loss but never zero weight | `HuberT::default()` or `HuberT::new(t)` |
| `norms::TukeyBiweight` | Redescending: residuals past the tuning constant get **zero** weight | `TukeyBiweight::default()` or `TukeyBiweight::new(c)` |
| `norms::AndrewWave` | Redescending sine-wave criterion | `AndrewWave::new(a)` |
| `norms::LeastSquares` | Ordinary squared loss (no robustness) | `LeastSquares` |

The extension module `norms_ext` adds `Hampel`, `RamsayE`, and `TrimmedMean`.

## Fitting an RLM

`Rlm::new(endog, exog, norm)` builds the model; `.fit()` returns an
`RlmResults`. As elsewhere, you supply any intercept column yourself.

```rust
use ndarray::{Array1, Array2};
use solow_robust::norms::{HuberT, TukeyBiweight};
use solow_robust::{Rlm, ScaleEst};

let xs: Vec<f64> = (1..=10).map(|i| i as f64).collect();
let y = Array1::from(vec![
    2.6, 3.1, 3.4, 4.1, 4.4, 5.1, 5.4, 6.1, 6.4, 100.0, // last is an outlier
]);
let exog =
    Array2::from_shape_fn((10, 2), |(i, j)| if j == 0 { 1.0 } else { xs[i] });

// Tukey's redescending biweight fully rejects the gross outlier.
let res = Rlm::new(y.clone(), exog.clone(), TukeyBiweight::default())
    .unwrap()
    .fit()
    .unwrap();
println!("biweight slope   = {:.4}", res.params[1]);
println!("weight of obs 10 = {:.2}", res.weights[9]); // 0.0 — fully rejected

// Huber's monotone criterion, with the MAD scale held fixed across iterations.
let huber = Rlm::new(y, exog, HuberT::default())
    .unwrap()
    .scale_est(ScaleEst::Mad)
    .fit()
    .unwrap();
println!("huber slope = {:.4}", huber.params[1]);
```

An ordinary least-squares fit would chase the outlier and report a slope near
10; the biweight fit stays close to the clean trend (≈ 0.5) and assigns the
outlying observation a weight of exactly zero.

## Results

`RlmResults` exposes:

- `params` — the robust coefficient estimates,
- `bse`, `tvalues`, `pvalues` — standard errors and normal-theory inference,
- `scale` — the final robust scale estimate `σ`,
- `weights` — the final IRLS weight on each observation (small or zero for
  outliers),
- `converged` — whether the IRLS loop met its tolerance.

## Tuning the estimator

`Rlm` is configured with a builder-style API. The defaults match the reference
(`scale_est = 'mad'`, `conv = 'dev'`, `update_scale = true`, `maxiter = 50`,
`tol = 1e-8`):

- `.scale_est(ScaleEst::Mad)` — choose the scale estimator (median absolute
  deviation by default; `HuberScale` is also available),
- the remaining knobs (`Conv`, `update_scale`) mirror the reference's RLM
  options.

[`solow-robust`]: https://github.com/solow-rs/solow
[`solow-regression`]: ./regression.md
