# Pipelines & hyperparameter search

`solow-pipeline` covers sequential and parallel composition of transformers and estimators, column-selective transforms, transformed-target wrappers, full-grid and randomized search, and successive-halving variants for cheap-then-expensive search.

## Composition

| Type | What it does |
|---|---|
| `Pipeline` | Sequential composition of transformers followed by a final estimator. Boxed closures for full flexibility. |
| `FeatureUnion` | Parallel-concat of multiple transformer outputs (feature-wise horizontal concatenation). |
| `ColumnTransformer` | Apply different transformers to different column subsets in one `fit`. |
| `TransformedTargetRegressor` | Wrap a regressor with a target transform and inverse transform (e.g. log-target regression). |

## Search

| Searcher | What it does |
|---|---|
| `GridSearchCV` | Full-factorial grid search over hyperparameter combinations on any splitter. |
| `RandomizedSearchCV` | Deterministic MMIX-LCG-driven random sampling over caller-supplied per-parameter closures. |
| `HalvingGridSearchCV` | Successive-halving over a grid: cheap early filter, expensive final rounds. |
| `HalvingRandomSearchCV` | Successive-halving with randomized sampling. |

All searchers return a `SearchResult` with mean and unbiased standard deviation of the per-fold scores.

## Example

```rust
use solow_pipeline::{Pipeline, GridSearchCV, ParamGrid};
use solow_preprocessing::StandardScaler;
use solow_discrete::LogisticRegressionCV;
use solow_cv::StratifiedKFold;

let pipe = Pipeline::new()
    .step("scale", Box::new(StandardScaler::new()))
    .step("clf",   Box::new(LogisticRegressionCV::new()));

let grid = ParamGrid::new().add("clf__C", vec![0.01, 0.1, 1.0, 10.0]);
let cv = StratifiedKFold::new(5)?;
let search = GridSearchCV::new(pipe, grid, cv).fit(&x, &y)?;
# Ok::<_, solow_core::Error>(())
```
