# solow-semi-supervised

**Semi-supervised learning for Rust.** Self-training around any probabilistic classifier, and graph-based label propagation and label spreading for partially-labeled data.

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-semi-supervised = "0.7"
```

## Quick example

```rust
use solow_semi_supervised::SelfTrainingClassifier;

// y_partial: labeled examples have their class; unlabeled have -1.
let clf = SelfTrainingClassifier::new(base_classifier)
    .threshold(0.75)
    .max_iter(10)
    .fit(&x, &y_partial)?;
```

## What is inside

| Estimator | What it does |
|---|---|
| `SelfTrainingClassifier` | Iterative self-labeling around any classifier that returns probabilities. Adds high-confidence unlabeled points to the training set each round. |
| `LabelPropagation` | Graph-based label propagation with RBF or k-NN affinity graph. Hard labels. |
| `LabelSpreading` | Zhou 2004 normalized-Laplacian label spreading. Softer than label propagation, robust to noisy labels. |

## Correctness

- Cross-verified against committed golden reference fixtures on every CI run.
- `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.
