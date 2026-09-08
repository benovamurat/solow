# Semi-supervised learning

`solow-semi-supervised` provides self-training around any probabilistic classifier and graph-based label propagation and label spreading for partially-labeled data.

## Estimators

| Estimator | What it does |
|---|---|
| `SelfTrainingClassifier` | Iterative self-labeling around any classifier that returns probabilities. Adds high-confidence unlabeled points to the training set each round. |
| `LabelPropagation` | Graph-based label propagation with RBF or k-NN affinity graph. Hard labels. |
| `LabelSpreading` | Zhou (2004) normalized-Laplacian label spreading. Softer than label propagation, robust to noisy labels. |

## Example

```rust
use solow_semi_supervised::SelfTrainingClassifier;

// y_partial: labeled examples have their class; unlabeled have -1.
let clf = SelfTrainingClassifier::new(base_classifier)
    .threshold(0.75)
    .max_iter(10)
    .fit(&x, &y_partial)?;
# Ok::<_, solow_core::Error>(())
```

## References

- Zhou, D., Bousquet, O., Lal, T. N., Weston, J., & Schölkopf, B. (2004). *Learning with local and global consistency*. NIPS.
