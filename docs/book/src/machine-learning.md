# Machine learning

Solow ships a dedicated machine-learning surface with the standard
`fit` / `transform` / `predict` estimator API across several focused
crates, plus penalised linear models in `solow-regression`. Every
estimator is Rust-native, zero-`unsafe`, and deterministic under a
caller-supplied seed.

## `solow-preprocessing` — features first

| Type | Purpose |
| --- | --- |
| `StandardScaler` | Zero-mean, unit-variance per column (Welford one-pass) |
| `MinMaxScaler` | Range projection onto an arbitrary `[a, b]` |
| `RobustScaler` | Median / IQR — outlier-immune |
| `MaxAbsScaler` | `x / max|x|`, sparsity-preserving |
| `Normalizer` | Per-row L¹ / L² / L∞ normalisation |
| `PolynomialFeatures` | Graded-lex monomials up to `degree`, with `interaction_only` and `include_bias` |
| `LabelEncoder` | 1-D label ↔ `usize` (lexicographic vocabulary) |
| `OrdinalEncoder` | Per-column `LabelEncoder` for a full matrix |
| `OneHotEncoder` | Dummy-variable expansion with optional `drop_first` |
| `KBinsDiscretizer` | Uniform / quantile / KMeans binning |

Every scaler exposes `fit`, `transform`, `fit_transform`, and
`inverse_transform`; the inverse round-trip is exact to machine
precision on well-scaled data.

## `solow-cluster` — unsupervised clustering

| Estimator | Notes |
| --- | --- |
| `KMeans` | Lloyd's algorithm; `k-means++` init by default; `n_init` restarts; deterministic under `seed` |
| `Dbscan` | Ester-Kriegel-Sander-Xu (KDD 1996) with `Core` / `Border` / `Noise` roles |
| `AgglomerativeClustering` | Bottom-up hierarchical merge with `Single` / `Complete` / `Average` / `Ward` linkage; reports the full dendrogram |

## `solow-neighbors` — nearest-neighbour data structures and estimators

| Type | Purpose |
| --- | --- |
| `KdTree` | Balanced k-dimensional tree with bounding-box pruning; deterministic build |
| `KNeighborsClassifier` | Majority-vote / distance-weighted classification |
| `KNeighborsRegressor` | Neighbour-average / distance-weighted regression |

Fit is `O(n log n)`; a single k-nearest query is `O(k log n)` expected
in low dimensions.

## `solow-regression` — penalised linear models

Ridge / Lasso / ElasticNet were added alongside the existing OLS /
WLS / GLS / GLSAR / QuantReg surface:

| Estimator | Objective | Solver |
| --- | --- | --- |
| `Ridge` | `‖y − Xβ‖² / (2n) + α · ‖β‖²` | Cholesky closed form |
| `Lasso` | `‖y − Xβ‖² / (2n) + α · ‖β‖₁` | Coordinate descent (Friedman-Hastie-Tibshirani 2010) |
| `ElasticNet` | `‖y − Xβ‖² / (2n) + α · (ρ·‖β‖₁ + ½(1−ρ)·‖β‖²)` | Coordinate descent |

All three take `fit_intercept: bool`; centering is handled internally
and the reported intercept corresponds to the raw-feature model.

## One-line pipeline

The umbrella `prelude` re-exports every new type, so a canonical
"scale → fit penalised model → predict → score" chain is one line
of `use`:

```rust
use solow::prelude::*;
use ndarray::array;

let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]];
let y = array![3.0, 5.0, 7.0, 9.0, 11.0];

let scaler = StandardScaler::fit(x.view())?;
let xs = scaler.transform(x.view())?;

let ridge = Ridge::fit(y.view(), xs.view(), 0.1, true)?;
let yhat = ridge.predict(xs.view())?;

let mse = mean_squared_error(y.view(), yhat.view(), None)?;
# Ok::<_, solow_core::Error>(())
```

## References

* Arthur, D., & Vassilvitskii, S. (2007). *k-means++: The advantages
  of careful seeding*. SODA '07, 1027-1035.
* Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996). *A density-
  based algorithm for discovering clusters in large spatial databases
  with noise.* KDD-96, 226-231.
* Friedman, J., Hastie, T., & Tibshirani, R. (2010). *Regularization
  paths for generalized linear models via coordinate descent.* JSS 33(1).
* Bentley, J. L. (1975). *Multidimensional binary search trees used
  for associative searching.* CACM 18(9), 509-517.
