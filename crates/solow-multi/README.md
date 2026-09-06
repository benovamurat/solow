# solow-multi

**Multi-class and multi-output meta-estimators for Rust.** One-vs-rest, one-vs-one, output-code, multi-output regressor and classifier, and chain estimators that model target correlations.

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-multi = "0.7"
```

## Quick example

```rust
use solow_multi::OneVsRestClassifier;

let ovr = OneVsRestClassifier::new(base_binary_classifier).fit(&x, &y)?;
let proba = ovr.predict_proba(&x_test)?;
```

## What is inside

### Multi-class

| Estimator | What it does |
|---|---|
| `OneVsRestClassifier` | Fit one binary classifier per class against the rest. |
| `OneVsOneClassifier` | Fit one binary classifier per class pair and vote. |
| `OutputCodeClassifier` | Error-correcting output codes with configurable code size and random codebook. |

### Multi-output

| Estimator | What it does |
|---|---|
| `MultiOutputRegressor` | Fit an independent regressor per output. |
| `MultiOutputClassifier` | Fit an independent classifier per output. |
| `RegressorChain` | Sequential regressor chain where each target uses previous target predictions. |
| `ClassifierChain` | Sequential classifier chain, modelling label correlations. |

## Correctness

- Cross-verified against committed golden reference fixtures on every CI run.
- `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.
