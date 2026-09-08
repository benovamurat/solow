# Decision trees

`solow-tree` provides CART decision trees for classification and regression with Gini, entropy, MSE, and MAE splitters, and Extra-Tree variants with fully-randomized thresholds. Deterministic tie-breaking. Arena-stored nodes.

## Estimators

| Estimator | What it does |
|---|---|
| `DecisionTreeClassifier` | CART classifier with `Gini` or `Entropy` splitters, class weights, deterministic tie-breaking. |
| `DecisionTreeRegressor` | CART regressor with `Mse` or `Mae` splitters. Uses running mean and variance updates for O(n log n) split search. |
| `ExtraTreeClassifier` | Fully-randomized classifier tree: chooses a random threshold rather than searching for the best. |
| `ExtraTreeRegressor` | Extra-random regression tree variant. |

## `TreeParams` controls

| Field | Purpose |
|---|---|
| `max_depth` | Maximum tree depth. `None` for unlimited. |
| `min_samples_split` | Minimum samples to split an internal node. |
| `min_samples_leaf` | Minimum samples per leaf. |
| `min_impurity_decrease` | Minimum impurity drop for a split to be accepted. |
| `max_leaf_nodes` | Best-first growth cap on leaves. |
| `max_features` | Number of features considered at each split. |
| `seed` | Deterministic MMIX-LCG for feature subsampling and tie-breaking. |

## Example

```rust
use solow_tree::{DecisionTreeRegressor, RegressionCriterion, TreeParams};

let tree = DecisionTreeRegressor::new(
    RegressionCriterion::Mse,
    TreeParams::new()
        .max_depth(Some(10))
        .min_samples_split(2)
        .seed(42),
).fit(&x, &y)?;

let pred = tree.predict(&x_test)?;
# Ok::<_, solow_core::Error>(())
```

## Correctness

Cross-verified against committed golden reference fixtures. Deterministic tie-breaking: identical inputs produce identical trees across runs and platforms under a fixed seed.
