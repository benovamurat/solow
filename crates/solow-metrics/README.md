# solow-metrics

**Model evaluation metrics for Rust.** Regression, classification, calibration, cluster, forecast, and Bayesian model-comparison metrics. Post-hoc probability calibrators. Distribution-free conformal prediction. Model-agnostic interpretability tools.

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-metrics = "0.7"
```

## Quick example

```rust
use solow_metrics::{r2_score, roc_auc_score, classification_report};

let r2 = r2_score(y_true.view(), y_pred.view(), None)?;
let auc = roc_auc_score(y_true.view(), y_score.view())?;
let report = classification_report(y_true.view(), y_pred.view(), None)?;
```

## What is inside

### Regression

MSE, RMSE, MAE, median AE, max error, R², explained variance, MAPE, sMAPE, MSLE, RMSLE, pinball, D² absolute, D² Tweedie, mean Poisson deviance, mean Gamma deviance, robust Huber and log-cosh losses.

### Classification

Confusion matrix, accuracy, balanced accuracy, precision, recall, F-beta (binary / macro / micro / weighted), Matthews correlation, Cohen kappa, hinge loss, binary and multiclass log loss, ROC curve, ROC-AUC (binary + OvR + Hand-Till OvO), precision-recall curve, average precision, top-k accuracy, binary and multiclass focal losses.

### Calibration

Reliability curve (uniform or quantile bins), ECE, MCE, top-1 ECE, multiclass Brier, ranked probability score, three-term Brier decomposition (reliability, resolution, uncertainty). Post-hoc calibrators: `PlattScaling`, `IsotonicRegression`, `TemperatureScaling`.

### Cluster evaluation

Silhouette, adjusted Rand, adjusted MI, normalized MI, homogeneity, completeness, v-measure, Fowlkes-Mallows, Calinski-Harabasz, Davies-Bouldin.

### Pairwise kernels and distances

`pairwise_distances`, `rbf_kernel`, `linear_kernel`, `polynomial_kernel`, `sigmoid_kernel`, `laplacian_kernel`, `cosine_similarity`, `chi2_kernel`.

### Forecast comparison

MASE, RMSSE, pinball, Winkler interval score, Harvey-Leybourne-Newbold `diebold_mariano`, `giacomini_white_test` (conditional predictive ability, generalization of DM).

### Model comparison

`friedman_test` with Iman-Davenport adjustment, `nemenyi_critical_difference`, `wilcoxon_signed_rank` with tie correction, WAIC, PSIS-LOO with per-observation Pareto-k̂ diagnostic.

### Effect sizes

Cohen's d, Hedges' g, Glass's delta, eta², omega², Cliff's delta, Cramer's V.

### Distribution-free intervals

`SplitConformal`, `JackknifePlus` (Barber-Candes-Ramdas-Tibshirani 2021).

### Interpretability

`permutation_importance` (Breiman-Fisher), `partial_dependence`, `accumulated_local_effects` (Apley-Zhu 2020).

### Reporting

`classification_report` renders a printable per-class precision / recall / F1 / support table.

## Numerical care

Core losses (`mean_squared_error`, `mean_absolute_error`, `r2_score`, `explained_variance_score`) use the Kahan / Neumaier compensated summation and Welford one-pass variance primitives from `solow-core::numeric`. Long sums of mixed-magnitude residuals do not lose low-order bits.

## `serde` support

Every public result and report struct derives `Serialize` and `Deserialize` under the opt-in `serde` feature.

```toml
[dependencies]
solow-metrics = { version = "0.7", features = ["serde"] }
```

An entire evaluation loop can then be persisted as JSON and diffed across runs, model versions, or CI jobs.

## Correctness

Every metric agrees with its canonical mathematical definition (or the published Diebold-Mariano / Harvey-Leybourne-Newbold / López de Prado / Vehtari-Gelman-Gabry reference) to machine precision on the committed fixture suite.

`#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.
