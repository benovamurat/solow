# Cross-validation, resampling, and bootstrap

`solow-cv` gives you the three composable layers of a modern
resampling pipeline: **splitters** that carve a sample into
train/test folds, **cross-validated scoring** that runs a `fit_and_score`
callback across the folds, and **bootstrap confidence intervals** for
any scalar statistic. Random splits and bootstrap resamples use a
portable 64-bit MMIX-LCG so a fixed seed produces bit-for-bit identical
folds and replicates across runs and platforms.

## Splitters

Every splitter emits `Vec<Split>` (`Split { train: Vec<usize>, test: Vec<usize> }`),
which plugs into the scoring helper below or into any manual iteration
loop.

| Splitter | Constructor | Notes |
| --- | --- | --- |
| K-fold | `KFold::new(k)?.shuffle(bool).seed(u64)` | Standard "first `n mod k` folds are one bigger" convention. |
| Stratified K-fold | `StratifiedKFold::new(k)?` + `.split(&y)` | Class-preserving. |
| Time series (walk-forward) | `TimeSeriesSplit::new(k)?.test_size(n).gap(n).max_train_size(n)` | Strictly forward-looking; `gap` and `max_train_size` for rolling windows. |
| Leave-one-out | `LeaveOneOut::new()` | `n` folds of size 1. |
| Shuffle-split | `ShuffleSplit::new(n_splits, test_size)?.seed(u64)` | Random `n`-times repeats. |
| Group K-fold | `GroupKFold::new(k)?` + `.split(&groups)` | Every group in exactly one fold (greedy longest-first). |
| Stratified Group K-fold | `StratifiedGroupKFold::new(k)?.split(&y, &groups)` | Group-aware + class-preserving. |
| Purged K-fold | `PurgedKFold::new(k, purge, embargo)?` | López de Prado leakage-safe walk-forward with symmetric purge and trailing embargo. |
| Combinatorial Purged K-fold | `CombinatorialPurgedKFold::new(k, n_test, purge, embargo)?` | Enumerates every `C(k, n_test)` subset of contiguous blocks; standard for low-variance Sharpe / drawdown back-tests. |

Row-count-only splitters implement the shared `Splitter` trait so
generic scorers can iterate over any of them. The group- and time-aware
splitters take an extra `groups` slice or the raw `n` argument; they
emit the same `Vec<Split>` shape.

```rust
use solow_cv::{KFold, Splitter};

let kf = KFold::new(5)?.shuffle(true).seed(42);
for split in kf.split(1000)? {
    let (train, test) = (split.train, split.test);
    // fit on train, score on test
}
# Ok::<_, solow_core::Error>(())
```

## `cross_val_score`

Runs a callback across every fold a splitter produces and returns the
per-fold scores plus their mean, unbiased standard deviation, and
standard error.

```rust
use solow_cv::{cross_val_score, KFold};
use solow_metrics::mean_squared_error;

let kf = KFold::new(5)?;
let scores = cross_val_score(&kf, y.len(), |train_idx, test_idx| {
    // fit a model on rows in train_idx, predict on test_idx, return the score
    let pred = my_fit_and_predict(train_idx, test_idx)?;
    let test_y = y.select(Axis(0), test_idx);
    Ok(-mean_squared_error(test_y.view(), pred.view(), None)?)
})?;

println!("Mean CV score: {:.4} ± {:.4}", scores.mean(), scores.standard_error());
```

`cross_val_score_from_folds(folds, fit_and_score)` is the variant that
takes a pre-computed fold list — useful when the same folds should be
shared across a hyperparameter sweep.

### Parallel folds

Behind the opt-in `parallel` feature, `cross_val_score_parallel` runs
folds on a rayon thread pool. The callback must be `Fn + Send + Sync`
(rather than `FnMut`), and the returned scores stay in fold order
regardless of worker completion order.

```toml
solow-cv = { version = "0.2", features = ["parallel"] }
```

```rust
use solow_cv::cross_val_score_parallel;
let scores = cross_val_score_parallel(&kf, y.len(), |train, test| { /* ... */ })?;
```

## Bootstrap confidence intervals

`bootstrap_ci` returns a point estimate together with a two-sided
confidence interval for any scalar statistic. The callback receives
resampled row indices — this keeps the statistic typed independently
of the underlying container.

| Method | Description |
| --- | --- |
| `BootstrapMethod::Percentile` | Classical `[Q_{α/2}, Q_{1-α/2}]` of the bootstrap distribution. |
| `BootstrapMethod::Basic` | Reverse-percentile `[2θ̂ - Q_{1-α/2}, 2θ̂ - Q_{α/2}]`. |
| `BootstrapMethod::Bca` | Efron's bias-corrected & accelerated, with a jackknife estimate of `â`. |

```rust
use solow_cv::{bootstrap_ci, BootstrapMethod};

let data: Vec<f64> = /* ... */;
let ci = bootstrap_ci(
    data.len(),
    |idx| Ok(idx.iter().map(|&i| data[i]).sum::<f64>() / idx.len() as f64),
    1000,                       // number of replicates
    0.95,                       // confidence
    BootstrapMethod::Bca,
    42,                          // seed
)?;

println!("mean ∈ [{:.4}, {:.4}] (SE = {:.4})",
    ci.low, ci.high, ci.standard_error());
```

## Serialization

`Split`, `CrossValScores`, and `BootstrapCi` derive `serde::Serialize` /
`serde::Deserialize` behind the opt-in `serde` feature, so a whole
resampling pipeline can be persisted as JSON:

```toml
solow-cv = { version = "0.2", features = ["serde"] }
```

## Module reference

- [`solow-cv`](https://docs.rs/solow-cv) — the full rustdoc.
