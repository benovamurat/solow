# Model evaluation metrics

`solow-metrics` bundles every number you report *after* an estimator has
been fit — losses and scores for regression, calibration diagnostics
for probabilistic classifiers, forecast-comparison tests, non-parametric
model comparison, Bayesian model comparison, and model-agnostic
interpretability tools. Every metric agrees with its canonical
definition (or the published reference paper) to machine precision on
the fixture suite in `tests/fixtures/metrics/*.json`.

The functions all take `ArrayView` inputs so they compose with slices,
subranges, and matrix rows without copies. Every metric returns a
`Result` — mismatched shapes, negative sample weights, unknown classes,
or a constant target for R² are user errors, not silent NaNs.

## Regression

| Metric | Call |
| --- | --- |
| Mean squared error | `mean_squared_error(y, ŷ, w?)` |
| Root mean squared error | `root_mean_squared_error(y, ŷ, w?)` |
| Mean absolute error | `mean_absolute_error(y, ŷ, w?)` |
| Median absolute error | `median_absolute_error(y, ŷ)` |
| Max absolute residual | `max_error(y, ŷ)` |
| MAPE / sMAPE | `mean_absolute_percentage_error(y, ŷ, w?)`, `symmetric_mean_absolute_percentage_error(y, ŷ, w?)` |
| Squared / root squared log error | `mean_squared_log_error(y, ŷ, w?)`, `root_mean_squared_log_error(y, ŷ, w?)` |
| R² | `r2_score(y, ŷ, w?)` |
| Explained variance | `explained_variance_score(y, ŷ, w?)` |
| Mean pinball loss (quantile) | `mean_pinball_loss(y, ŷ, α, w?)` |
| D² absolute / D² Tweedie | `d2_absolute_error_score(y, ŷ, w?)`, `d2_tweedie_score(y, ŷ, w?, power)` |
| Huber loss | `huber_loss(y, ŷ, δ, w?)` |
| Log-cosh loss | `log_cosh_loss(y, ŷ, w?)` |
| One-call bundle | `RegressionReport::compute(y, ŷ, w?)` |

Core sums use the Kahan/Neumaier compensated accumulator from
`solow-core::numeric`, so long residual sums of mixed magnitudes do
not lose low-order bits.

## GLM / Tweedie deviance

| Metric | Call |
| --- | --- |
| Tweedie deviance (any power) | `mean_tweedie_deviance(y, ŷ, w?, power)` |
| Poisson deviance | `mean_poisson_deviance(y, ŷ, w?)` |
| Gamma deviance | `mean_gamma_deviance(y, ŷ, w?)` |
| D² (deviance-based) | `d2_tweedie_score(y, ŷ, w?, power)` |

These are the natural goodness-of-fit numbers to pair with a
`solow-glm::Glm` fitted on Poisson or Gamma data.

## Classification

Labels are `usize` class indices in `[0, k)`. Probabilities live in a
`(n, k)` matrix that must sum to one row-wise for probability metrics.

| Group | Calls |
| --- | --- |
| Confusion matrix | `confusion_matrix(y, ŷ, num_classes?)` |
| Accuracy family | `accuracy_score`, `zero_one_loss`, `balanced_accuracy_score` |
| Precision / Recall / Fβ | `precision_recall_fscore(y, ŷ, Average, β, w?)`, `fbeta_score(...)` |
| Averaging | `Average::Binary` / `Macro` / `Weighted` / `Micro` |
| Multi-class agreement | `matthews_corrcoef`, `cohen_kappa_score(y, ŷ, KappaWeights)` |
| Log-loss family | `binary_log_loss(y, p, w?, ε)`, `log_loss(y, p, w?, ε)` |
| Focal losses | `binary_focal_loss(y, p, γ, α, w?)`, `multiclass_focal_loss(y, p, γ, α, w?)` |
| Score / margin | `hinge_loss(y, decision, w?)`, `brier_score_loss(y, p, w?)` |
| ROC / PR | `roc_curve(y, score)`, `roc_auc_score(y, score)`, `precision_recall_curve(y, score)`, `average_precision_score(y, score)` |
| Multiclass ROC-AUC | `roc_auc_ovr_score(y, p, MulticlassAuc::{Macro,Weighted})`, `roc_auc_ovo_score(...)` (Hand-Till) |
| Top-k | `top_k_accuracy_score(y, p, k)` |
| Multiclass Brier / RPS | `multiclass_brier_score(y, p)`, `ranked_probability_score(y, p)` |
| Top-1 ECE | `top_label_calibration_error(y, p, n_bins)` |

## Calibration diagnostics and post-hoc calibrators

**Diagnostics** — how well a probability *means* what it says:

| Diagnostic | Call |
| --- | --- |
| Reliability curve | `reliability_curve(y, p, n_bins, BinStrategy::{Uniform,Quantile})` |
| Expected / Maximum CE | `expected_calibration_error(...)`, `maximum_calibration_error(...)` |
| Bröcker decomposition | `brier_decomposition(y, p, n_bins, strategy)` — reliability + resolution + uncertainty + within-bin variance + raw Brier |

**Calibrators** — train on held-out scores and re-map:

| Calibrator | Fit | Transform |
| --- | --- | --- |
| Platt (1999) | `PlattScaling::fit(s, y)` | `.transform(s)` |
| Isotonic (PAV) | `IsotonicRegression::fit(s, y)` | `.transform(s)` |
| Temperature (GPSW 2017) | `TemperatureScaling::fit(logits, y)` | `.transform(logits)` |

## Forecasting

| Metric | Call |
| --- | --- |
| MASE | `mase(y_true, y_pred, y_train, m)` |
| RMSSE | `rmsse(y_true, y_pred, y_train, m)` |
| Pinball loss | `pinball_loss(y, ŷ, α)` |
| Interval coverage | `interval_coverage(y, lo, hi)` |
| Mean Winkler interval score | `mean_interval_score(y, lo, hi, α)` |
| Diebold-Mariano (HLN-corrected) | `diebold_mariano(y, f1, f2, h, DmLoss)` |
| Giacomini-White conditional | `giacomini_white_test(y, f1, f2, test_regressors, h, DmLoss)` |

## Distribution-free prediction intervals (conformal)

Both give finite-sample coverage guarantees under exchangeability, no
distributional assumption required.

| Method | Fit | Interval |
| --- | --- | --- |
| Split conformal | `SplitConformal::fit(residuals, α)` | `.interval(ŷ)` → `PredictionInterval` |
| Jackknife+ (BCRT 2021) | `JackknifePlus::new(loo_residuals, α)` | `.interval(loo_predictions)` |

## Model comparison

**Non-parametric** — the Demšar (2006) recipe on an `(m × k)` score matrix
(datasets × models, higher-is-better):

- `friedman_test(scores)` → Iman-Davenport F-adjusted χ² with mean
  ranks and an F(k−1, (k−1)(m−1)) p-value.
- `nemenyi_critical_difference(k, m, α)` → tabulated critical distance
  at α ∈ {0.05, 0.10}.
- `wilcoxon_signed_rank(a, b)` → paired, tie-corrected normal
  approximation, scipy-compatible.

**Bayesian** — from a `(n_samples × n_observations)` posterior
log-likelihood matrix:

- `waic(log_lik)` → `WaicResult { elpd, elpd_se, p_waic, waic, pointwise }`.
- `psis_loo(log_lik)` → `PsisLooResult { elpd, elpd_se, p_loo, looic, pointwise, pareto_k }`.
  The per-observation Pareto tail-shape `k̂` is the standard reliability
  flag — anything `> 0.7` marks an LOO estimate that should be refit.

## Model-agnostic interpretability

| Diagnostic | Call | Returns |
| --- | --- | --- |
| Permutation importance | `permutation_importance(x, scorer, n_repeats, seed)` | `Vec<FeatureImportance>` |
| Partial dependence | `partial_dependence(x, feature, grid, predictor)` | `PartialDependence` |
| Accumulated local effects | `accumulated_local_effects(x, feature, n_bins, predictor)` | `AccumulatedLocalEffects` |

The estimator is decoupled: pass any `Fn(ArrayView2) -> Result<f64>`
scorer / `Fn(ArrayView2) -> Result<Array1<f64>>` predictor and this
layer never sees the underlying model.

## Serialization

Every public result / report struct derives `serde::Serialize` /
`serde::Deserialize` behind the opt-in `serde` feature:

```toml
solow-metrics = { version = "0.2", features = ["serde"] }
```

A full evaluation loop can then be persisted as JSON and diffed across
runs, model versions, or CI jobs.

## Reference-fixture pipeline

`tests/fixtures/metrics/*.json` holds committed golden reference
outputs. The Rust replay tests re-verify each fixture on every CI run
and fail on drift, guaranteeing the shipped metrics stay bit-wise
identical to their canonical mathematical definition.

## Module reference

- [`solow-metrics`](https://docs.rs/solow-metrics) — the full rustdoc.
