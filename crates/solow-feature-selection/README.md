# solow-feature-selection

**Feature selection for Rust.** Variance thresholding, univariate ANOVA F and mutual-information scoring, recursive feature elimination, and sequential (forward or backward) selection around any ranker.

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-feature-selection = "0.7"
```

## Quick example

```rust
use solow_feature_selection::{SelectKBest, score_f_classif};

let sel = SelectKBest::new(10).score_fn(score_f_classif).fit(&x, &y)?;
let x_kbest = sel.transform(&x)?;
```

## What is inside

### Univariate

| Selector | What it does |
|---|---|
| `VarianceThreshold` | Drop features with variance below a threshold (near-constant columns). |
| `SelectKBest` | Keep the top `k` features under a caller scoring function. |
| `SelectPercentile` | Keep the top `p%` of features. |
| `SelectFpr` | Keep features whose p-value is below `alpha` (false-positive rate control). |
| `SelectFdr` | Benjamini-Hochberg false-discovery-rate control. |
| `SelectFwe` | Bonferroni-corrected family-wise error control. |

### Scoring functions

| Function | What it does |
|---|---|
| `score_f_classif` | ANOVA F-statistic against class labels. |
| `score_f_regression` | Correlation-derived F-statistic against a continuous target. |

### Multivariate

| Selector | What it does |
|---|---|
| `Rfe` | Guyon-Weston-Barnhill-Vapnik 2002 recursive feature elimination around any caller-supplied ranker. |
| `SequentialFeatureSelector` | Greedy forward or backward selection with any scorer. |

## Correctness

- Cross-verified against committed golden reference fixtures on every CI run.
- `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.
