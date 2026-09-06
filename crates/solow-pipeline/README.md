# solow-pipeline

**Machine-learning pipelines and hyperparameter search for Rust.** Sequential and parallel composition of transformers and estimators, column-selective transforms, transformed-target wrappers, full-grid and randomized search, plus successive-halving variants for cheap-then-expensive search.

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-pipeline = "0.7"
```

## Quick example

```rust
use solow_pipeline::{Pipeline, GridSearchCV, ParamGrid};

let pipe = Pipeline::new()
    .step("scale", Box::new(StandardScaler::new()))
    .step("clf", Box::new(LogisticRegression::new()));

let grid = ParamGrid::new().add("clf__C", vec![0.01, 0.1, 1.0, 10.0]);
let search = GridSearchCV::new(pipe, grid, cv).fit(&x, &y)?;
```

## What is inside

### Composition

| Type | What it does |
|---|---|
| `Pipeline` | Sequential composition of transformers followed by a final estimator. Boxed closures for full flexibility. |
| `FeatureUnion` | Parallel-concat of multiple transformer outputs (feature-wise horizontal concatenation). |
| `ColumnTransformer` | Apply different transformers to different column subsets in one `fit`. |
| `TransformedTargetRegressor` | Wrap a regressor with a target transform and inverse transform (e.g. log-target regression). |

### Search

| Searcher | What it does |
|---|---|
| `GridSearchCV` | Full-factorial grid search over hyperparameter combinations on any splitter. |
| `RandomizedSearchCV` | Deterministic MMIX-LCG-driven random sampling over caller-supplied per-parameter closures. |
| `HalvingGridSearchCV` | Successive-halving over a grid: cheap early filter, expensive final rounds. |
| `HalvingRandomSearchCV` | Successive-halving with randomized sampling. |

All searchers return a full `SearchResult` grid with mean and unbiased standard deviation of the per-fold scores.

## Correctness

- Every searcher is deterministic under a fixed seed.
- Cross-verified against committed golden reference fixtures on every CI run.
- `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.
