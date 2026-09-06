# solow-ensemble

**Ensemble methods for Rust.** Random forests, gradient boosting (both classical and histogram-binned), bagging, boosting, voting, stacking, isolation forest, in one memory-safe pure-Rust crate. Every ensemble is deterministic under a caller seed.

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-ensemble = "0.7"
```

## Quick example

```rust
use solow_ensemble::RandomForestClassifier;

let rf = RandomForestClassifier::new()
    .n_estimators(200)
    .max_depth(Some(20))
    .max_features_sqrt()
    .seed(42)
    .fit(&x, &y)?;

let proba = rf.predict_proba(&x_test)?;
```

## What is inside

### Random forests

| Estimator | What it does |
|---|---|
| `RandomForestClassifier`, `RandomForestRegressor` | Breiman 2001 bagged CART forests with per-tree feature subsampling. |
| `ExtraTreesClassifier`, `ExtraTreesRegressor` | Fully-randomized forests: random splits without threshold search, faster to fit. |

### Gradient boosting

| Estimator | What it does |
|---|---|
| `GradientBoostingRegressor` | Friedman 2001 stagewise least-squares boosting with shrinkage and row subsampling. |
| `GradientBoostingClassifier` | Binary logistic gradient boosting with cross-entropy loss. |
| `HistGradientBoostingRegressor`, `HistGradientBoostingClassifier` | Histogram-binned boosting with per-feature 256-bin quantization for large-data speed. |

### Bagging and boosting meta-ensembles

| Estimator | What it does |
|---|---|
| `BaggingClassifier`, `BaggingRegressor` | Meta-ensembles over any base estimator with feature and sample bootstrap. |
| `AdaBoostClassifier` | Freund-Schapire SAMME multi-class AdaBoost. |
| `AdaBoostRegressor` | Drucker 1997 AdaBoost.R2 for regression. |

### Combiners

| Estimator | What it does |
|---|---|
| `VotingClassifier`, `VotingRegressor` | Soft or hard voting across a heterogeneous set of base learners. |
| `StackingClassifier`, `StackingRegressor` | Two-level stacking with an out-of-fold meta-learner. |

### Anomaly detection

| Estimator | What it does |
|---|---|
| `IsolationForest` | Liu-Ting-Zhou 2008 unsupervised anomaly scoring with normalised path length in `[0, 1]`. |

## Correctness

- Cross-verified against committed golden reference fixtures on every CI run.
- Every ensemble uses a portable MMIX-LCG PRNG so a fixed seed reproduces the fit bit-for-bit across runs and platforms.
- `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.
