# solow-calibration

**Probability calibration for classifiers.** Cross-validated post-hoc calibration with Platt sigmoid or isotonic regression, in one memory-safe pure-Rust crate.

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-calibration = "0.7"
```

## Quick example

```rust
use solow_calibration::{CalibratedClassifierCV, Method};

let cal = CalibratedClassifierCV::new(base_estimator, Method::Sigmoid)
    .cv(5)
    .fit(&x, &y)?;

let proba = cal.predict_proba(&x_test)?;
```

## What is inside

| Estimator | What it does |
|---|---|
| `CalibratedClassifierCV` | Cross-validated post-hoc probability calibration around any base classifier. Fits `k` base models on `k-1`-fold splits, then calibrates on the held-out fold and averages calibrators. |
| `Method::Sigmoid` | Platt scaling: parametric sigmoid fit by Newton-Raphson with safe target labels. |
| `Method::Isotonic` | Non-parametric monotone calibration via pool-adjacent-violators. |

## Correctness

- Cross-verified against committed golden reference fixtures on every CI run.
- `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.
