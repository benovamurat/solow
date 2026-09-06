# solow-tree

**Decision trees for Rust.** CART for classification and regression with Gini, entropy, MSE, and MAE splitters, plus Extra-Tree variants with fully-randomized thresholds. Deterministic tie-breaking. Arena-stored nodes.

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-tree = "0.7"
```

## Quick example

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
```

## What is inside

| Estimator | What it does |
|---|---|
| `DecisionTreeClassifier` | CART classifier with `Gini` or `Entropy` splitters, class weights, and deterministic tie-breaking. |
| `DecisionTreeRegressor` | CART regressor with `Mse` or `Mae` splitters. Uses running mean and variance updates for O(n log n) split search. |
| `ExtraTreeClassifier` | Fully-randomized classifier tree: chooses a random threshold rather than searching for the best. |
| `ExtraTreeRegressor` | Extra-random regression tree variant. |

### `TreeParams` controls

- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `min_impurity_decrease`
- `max_leaf_nodes`
- `max_features`
- `seed` (deterministic MMIX-LCG for feature subsampling and tie-breaking)

## Correctness

- Cross-verified against committed golden reference fixtures.
- Deterministic tie-breaking: identical inputs produce identical trees across runs and platforms under a fixed seed.
- `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.
