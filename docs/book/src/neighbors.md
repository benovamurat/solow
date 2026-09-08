# Nearest neighbors

`solow-neighbors` covers the full nearest-neighbor stack: data structures, k-NN and radius-neighbor predictors, nearest centroid, local outlier factor, and multivariate kernel density estimation.

## Data structures

| Type | What it does |
|---|---|
| `KdTree` | Balanced KD-tree with bounding-box pruning. Expected `O(k log n)` queries in low dimensions. |
| `BallTree` | Ball-tree partition for moderate-dimensional data where KD-tree pruning weakens. |

## Predictors

| Estimator | What it does |
|---|---|
| `KNeighborsClassifier` | k-NN classification with uniform or distance-weighted votes. |
| `KNeighborsRegressor` | k-NN regression with uniform or distance-weighted averaging. |
| `RadiusNeighborsClassifier` | Fixed-radius neighborhood classification. |
| `RadiusNeighborsRegressor` | Fixed-radius neighborhood regression. |
| `NearestCentroid` | Rocchio-style classifier that assigns each query to the nearest class-mean centroid. |

## Density and anomaly

| Estimator | What it does |
|---|---|
| `LocalOutlierFactor` | Breunig-Kriegel (2000) density-based outlier scoring. |
| `KernelDensity` | Multivariate KDE with Gaussian, tophat, Epanechnikov, exponential, linear, cosine kernels. |

## Example

```rust
use solow_neighbors::{KNeighborsClassifier, WeightKind};

let knn = KNeighborsClassifier::new(5)
    .weights(WeightKind::Distance)
    .fit(&x, &y)?;

let proba = knn.predict_proba(&x_test)?;
# Ok::<_, solow_core::Error>(())
```
