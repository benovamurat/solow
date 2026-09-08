# Feature selection

`solow-feature-selection` covers univariate selection with ANOVA F and mutual-information scoring, variance thresholding, recursive feature elimination, and sequential (forward or backward) selection around any ranker.

## Univariate selection

| Selector | What it does |
|---|---|
| `VarianceThreshold` | Drop features with variance below a threshold (near-constant columns). |
| `SelectKBest` | Keep the top `k` features under a caller scoring function. |
| `SelectPercentile` | Keep the top `p%` of features. |
| `SelectFpr` | Keep features whose p-value is below `alpha` (false-positive rate control). |
| `SelectFdr` | Benjamini-Hochberg false-discovery-rate control. |
| `SelectFwe` | Bonferroni-corrected family-wise error control. |

## Scoring functions

| Function | What it does |
|---|---|
| `score_f_classif` | ANOVA F-statistic against class labels. |
| `score_f_regression` | Correlation-derived F-statistic against a continuous target. |

## Multivariate selection

| Selector | What it does |
|---|---|
| `Rfe` | Guyon-Weston-Barnhill-Vapnik (2002) recursive feature elimination around any caller-supplied ranker. |
| `SequentialFeatureSelector` | Greedy forward or backward selection with any scorer. |

## Example

```rust
use solow_feature_selection::{SelectKBest, score_f_classif};

let sel = SelectKBest::new(10).score_fn(score_f_classif).fit(&x, &y)?;
let x_kbest = sel.transform(&x)?;
# Ok::<_, solow_core::Error>(())
```

## References

- Guyon, I., Weston, J., Barnhill, S., & Vapnik, V. (2002). *Gene selection for cancer classification using support vector machines*. Machine Learning 46, 389-422.
