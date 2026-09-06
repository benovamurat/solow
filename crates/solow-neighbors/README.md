# solow-neighbors

**Nearest-neighbor structures and estimators for Rust.** KD-tree, ball tree, k-NN classifier and regressor, radius-neighbors variants, nearest centroid, local outlier factor, and multivariate kernel density estimation.

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-neighbors = "0.7"
```

## Quick example

```rust
use solow_neighbors::{KNeighborsClassifier, WeightKind};

let knn = KNeighborsClassifier::new(5)
    .weights(WeightKind::Distance)
    .fit(&x, &y)?;

let proba = knn.predict_proba(&x_test)?;
```

## What is inside

### Data structures

| Type | What it does |
|---|---|
| `KdTree` | Balanced KD-tree with bounding-box pruning. Expected `O(k log n)` queries in low dimensions. |
| `BallTree` | Ball-tree partition for moderate-dimensional data where KD-tree pruning weakens. |

### Predictors

| Estimator | What it does |
|---|---|
| `KNeighborsClassifier` | k-NN classification with uniform or distance-weighted votes. |
| `KNeighborsRegressor` | k-NN regression with uniform or distance-weighted averaging. |
| `RadiusNeighborsClassifier` | Fixed-radius neighborhood classification. |
| `RadiusNeighborsRegressor` | Fixed-radius neighborhood regression. |
| `NearestCentroid` | Rocchio-style classifier that assigns each query to the nearest class-mean centroid. |

### Anomaly and density

| Estimator | What it does |
|---|---|
| `LocalOutlierFactor` | Breunig-Kriegel 2000 density-based outlier scoring. |
| `KernelDensity` | Multivariate kernel density estimation with Gaussian, tophat, Epanechnikov, exponential, linear, cosine kernels. |

## Correctness

- Cross-verified against committed golden reference fixtures on every CI run.
- `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.
