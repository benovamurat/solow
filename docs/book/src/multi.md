# Multi-class & multi-output

`solow-multi` covers one-vs-rest, one-vs-one, output-code, multi-output regressor and classifier, and chain estimators that model target correlations.

## Multi-class

| Estimator | What it does |
|---|---|
| `OneVsRestClassifier` | Fit one binary classifier per class against the rest. |
| `OneVsOneClassifier` | Fit one binary classifier per class pair and vote. |
| `OutputCodeClassifier` | Error-correcting output codes with configurable code size and random codebook. |

## Multi-output

| Estimator | What it does |
|---|---|
| `MultiOutputRegressor` | Fit an independent regressor per output. |
| `MultiOutputClassifier` | Fit an independent classifier per output. |
| `RegressorChain` | Sequential regressor chain where each target uses previous target predictions. |
| `ClassifierChain` | Sequential classifier chain, modelling label correlations. |

## Example

```rust
use solow_multi::OneVsRestClassifier;

let ovr = OneVsRestClassifier::new(base_binary_classifier).fit(&x, &y)?;
let proba = ovr.predict_proba(&x_test)?;
# Ok::<_, solow_core::Error>(())
```
