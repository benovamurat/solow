# Ensemble methods

`solow-ensemble` ships every mainstream ensemble family: random forests, gradient boosting (both classical and histogram-binned), bagging, boosting, voting, stacking, and isolation forest. Every ensemble is deterministic under a caller seed via a portable MMIX-LCG PRNG.

## Random forests and extra trees

| Estimator | What it does |
|---|---|
| `RandomForestClassifier`, `RandomForestRegressor` | Breiman (2001) bagged CART forests with per-tree feature subsampling. |
| `ExtraTreesClassifier`, `ExtraTreesRegressor` | Fully-randomized forests: random splits without threshold search, faster to fit. |

## Gradient boosting

| Estimator | What it does |
|---|---|
| `GradientBoostingRegressor` | Friedman (2001) stagewise least-squares boosting with shrinkage and row subsampling. |
| `GradientBoostingClassifier` | Binary logistic gradient boosting with cross-entropy loss. |
| `HistGradientBoostingRegressor`, `HistGradientBoostingClassifier` | Histogram-binned boosting with per-feature 256-bin quantization for large-data speed. |

## Bagging and boosting meta-ensembles

| Estimator | What it does |
|---|---|
| `BaggingClassifier`, `BaggingRegressor` | Meta-ensembles over any base estimator with feature and sample bootstrap. |
| `AdaBoostClassifier` | Freund-Schapire SAMME multi-class AdaBoost. |
| `AdaBoostRegressor` | Drucker (1997) AdaBoost.R2. |

## Combiners

| Estimator | What it does |
|---|---|
| `VotingClassifier`, `VotingRegressor` | Soft or hard voting across a heterogeneous set of base learners. |
| `StackingClassifier`, `StackingRegressor` | Two-level stacking with an out-of-fold meta-learner. |

## Anomaly detection

| Estimator | What it does |
|---|---|
| `IsolationForest` | Liu-Ting-Zhou (2008) unsupervised anomaly scoring with normalised path length in `[0, 1]`. |

## Example

```rust
use solow_ensemble::RandomForestClassifier;

let rf = RandomForestClassifier::new()
    .n_estimators(200)
    .max_depth(Some(20))
    .max_features_sqrt()
    .seed(42)
    .fit(&x, &y)?;

let proba = rf.predict_proba(&x_test)?;
# Ok::<_, solow_core::Error>(())
```

## References

- Breiman, L. (2001). *Random forests*. Machine Learning 45, 5-32.
- Friedman, J. H. (2001). *Greedy function approximation: A gradient boosting machine*. Annals of Statistics 29(5), 1189-1232.
- Freund, Y., & Schapire, R. E. (1997). *A decision-theoretic generalization of on-line learning and an application to boosting*. JCSS 55(1), 119-139.
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). *Isolation forest*. ICDM.
