# solow-cv

**Cross-validation and resampling for Rust.** Deterministic K-fold, stratified, group-aware, time-series, and leakage-safe purged splitters. Learning / validation / permutation curves. Bootstrap confidence intervals. Time-series block bootstrap. Optional rayon-parallel fold evaluation.

Part of the [Solow](https://crates.io/crates/solow) statistics and machine learning stack.

## Install

```toml
[dependencies]
solow-cv = "0.7"
```

## Quick example

```rust
use solow_cv::{StratifiedKFold, cross_val_score};

let cv = StratifiedKFold::new(5)?.shuffle(true).seed(42);
let scores = cross_val_score(&cv, n, |train, test| {
    let m = fit_your_model(&x, &y, train)?;
    Ok(m.score(&x, &y, test)?)
})?;
println!("{:.4} +/- {:.4}", scores.mean, scores.std);
```

## What is inside

### Random and stratified splitters

| Splitter | What it does |
|---|---|
| `KFold` | K-fold with optional shuffle and seed. Standard "first `n mod k` folds are one bigger" convention. |
| `StratifiedKFold` | K-fold preserving per-fold class prevalence. |
| `RepeatedKFold`, `RepeatedStratifiedKFold` | Multiple K-fold repetitions with different seeds. |
| `ShuffleSplit`, `StratifiedShuffleSplit` | Random `n`-times splits with configurable test size. |
| `LeaveOneOut` | n folds of size 1. |
| `LeavePOut` | All `C(n, p)` folds of size p. |

### Group-aware splitters

| Splitter | What it does |
|---|---|
| `GroupKFold` | Every group in exactly one fold, via greedy longest-first balancer. |
| `StratifiedGroupKFold` | GroupKFold that also preserves class prevalence via a round-robin seed and SSE minimisation. |
| `GroupShuffleSplit` | Group-aware random splits. |

### Time series

| Splitter | What it does |
|---|---|
| `TimeSeriesSplit` | Walk-forward validation with configurable `test_size`, `gap`, and `max_train_size` for rolling windows. |
| `PurgedKFold` | López de Prado (2018) walk-forward K-fold with symmetric purge band around every test block and optional trailing embargo. |
| `CombinatorialPurgedKFold` | CPCV that enumerates every `C(n_splits, n_test)` subset of contiguous blocks, giving a much richer set of distinct back-test paths than plain K-fold. |

### Scoring and curves

| Function | What it does |
|---|---|
| `cross_val_score` | Run any fit-and-score callback across every fold. Returns per-fold scores plus mean, unbiased standard deviation, and standard error. |
| `cross_val_score_parallel` | Same, but evaluates folds concurrently on a rayon thread pool (opt-in `parallel` feature). |
| `learning_curve` | Training-size sweep for diagnosing bias / variance. |
| `validation_curve` | Hyperparameter sweep for regularization curves. |
| `permutation_test_score` | Non-parametric significance test for a cross-val score. |

### Bootstrap

| Function | What it does |
|---|---|
| `bootstrap_ci` | Percentile, basic ("reverse-percentile"), and Efron BCa confidence intervals via jackknife acceleration. All three computed from the same replicates. |
| `moving_block_bootstrap_indices` | Overlapping blocks of fixed length for time-series resampling. |
| `circular_block_bootstrap_indices` | Wrap-around blocks that avoid the end-of-series bias. |
| `stationary_bootstrap_indices` | Politis-Romano stationary bootstrap with geometric block lengths. |

## Determinism

Random splits and bootstrap resamples use a seeded MMIX-constants 64-bit LCG. Folds and replicates are bit-for-bit reproducible across runs and platforms.

## `serde` and `parallel` features

```toml
[dependencies]
solow-cv = { version = "0.7", features = ["serde", "parallel"] }
```

## Correctness

- Every splitter is deterministic under a fixed seed.
- Cross-verified against committed golden reference fixtures.
- `#![forbid(unsafe_code)]`.

## License

BSD-3-Clause. See the [Solow workspace](https://github.com/benovamurat/solow) for the full stack.
