# solow-metrics

Model-evaluation metrics for the Solow statistical stack — the numbers you
report after a fit, not the ones an estimator uses internally.

## Coverage

- **Regression** — MSE, RMSE, MAE, median AE, max error, MAPE, sMAPE, MSLE,
  RMSLE, R², explained variance, mean pinball loss, D² (absolute and Tweedie),
  Huber and log-cosh robust losses, and a `RegressionReport` bundle.
- **GLM / Tweedie deviance** — `mean_tweedie_deviance`, `mean_poisson_deviance`,
  `mean_gamma_deviance`, `d2_tweedie_score` — the natural goodness-of-fit
  scores for a fitted Poisson or Gamma GLM.
- **Classification** — confusion matrix, accuracy, balanced accuracy,
  precision / recall / Fβ with `Binary` / `Macro` / `Micro` / `Weighted`
  averaging, Matthews correlation, Cohen's κ (linear and quadratic), binary
  and multiclass log-loss, Brier score, hinge loss, binary ROC / PR curves
  and AUC / AP, one-vs-rest and Hand-Till one-vs-one multiclass ROC-AUC,
  multiclass Brier, ranked probability score, top-1 ECE, and binary /
  multiclass focal loss.
- **Calibration** — reliability curves (uniform or quantile bins), ECE, MCE,
  Sanders/Murphy Brier decomposition with within-bin dispersion, and post-hoc
  calibrators (`PlattScaling`, `IsotonicRegression`, `TemperatureScaling`).
- **Conformal prediction** — distribution-free prediction intervals with
  finite-sample coverage guarantees: `SplitConformal` (Vovk-Gammerman-Shafer)
  and `JackknifePlus` (Barber-Candes-Ramdas-Tibshirani 2021).
- **Forecasting** — MASE, RMSSE, pinball loss, interval coverage, Winkler
  score, plus the Harvey-Leybourne-Newbold small-sample-corrected
  Diebold-Mariano test and the Giacomini-White conditional predictive
  ability test.
- **Model comparison** — non-parametric (Friedman + Nemenyi + Wilcoxon
  signed-rank) and Bayesian (WAIC + PSIS-LOO with Pareto-`k̂` diagnostic).
- **Interpretability** — permutation importance, partial dependence,
  accumulated local effects (Apley & Zhu 2020).

Every metric agrees with its canonical definition (or the published
Diebold-Mariano / Harvey-Leybourne-Newbold / López de Prado reference)
to machine precision on the fixture suite.

## Numerical care

The core losses (`mean_squared_error`, `mean_absolute_error`, `r2_score`,
`explained_variance_score`) use the Kahan / Neumaier compensated summation
and Welford one-pass variance primitives shipped in
[`solow-core::numeric`](https://docs.rs/solow-core), so long sums over
residuals of mixed magnitudes don't lose low-order bits.

## Features

- `serde` — derives `Serialize` / `Deserialize` on every public result and
  report struct so an entire evaluation loop can be persisted as JSON.

## Example

```rust
use ndarray::array;
use solow_metrics::{mean_squared_error, r2_score};

let y_true = array![3.0, -0.5, 2.0, 7.0];
let y_pred = array![2.5, 0.0, 2.0, 8.0];

let mse = mean_squared_error(y_true.view(), y_pred.view(), None).unwrap();
let r2  = r2_score(y_true.view(), y_pred.view(), None).unwrap();

assert!((mse - 0.375).abs() < 1e-12);
assert!((r2 - (1.0 - 1.5 / 29.1875)).abs() < 1e-10);
```

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
