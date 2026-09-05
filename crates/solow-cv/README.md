# solow-cv

Cross-validation, resampling, and bootstrap for the Solow statistical stack.

## Coverage

- **Splitters** — `KFold`, `StratifiedKFold`, `TimeSeriesSplit` (with
  `test_size`, `gap`, `max_train_size` for rolling-window walk-forward),
  `LeaveOneOut`, and seeded `ShuffleSplit` behind a shared `Splitter`
  trait.
- **Group-aware and leakage-safe splitters** — `GroupKFold` and
  `StratifiedGroupKFold` (no group is ever split across a fold),
  `PurgedKFold` (walk-forward + purge + embargo, López de Prado 2018),
  and `CombinatorialPurgedKFold` (every C(n_splits, n_test) block subset
  as a test fold — the standard leakage-safe splitter for financial
  back-tests).
- **Cross-validated scoring** — `cross_val_score(splitter, n,
  fit_and_score)` and `cross_val_score_from_folds(...)`. Returns
  `CrossValScores { scores, n_folds }` with `mean()`, `std()`, and
  `standard_error()`.
- **Parallel evaluation** (opt-in `parallel` feature) —
  `cross_val_score_parallel(...)` runs every fold on a rayon thread
  pool.
- **Bootstrap confidence intervals** — `bootstrap_ci(n, statistic,
  n_boot, confidence, method, seed)` with the classical percentile
  method, the reverse-percentile ("basic") method, and Efron's
  bias-corrected and accelerated (BCa) method using a jackknife
  acceleration estimate.

## Determinism

Random splits and bootstrap resamples use a portable 64-bit MMIX-LCG,
seeded by the caller. A fixed seed produces bit-for-bit identical folds
and replicates across runs and platforms.

## Features

- `parallel` — enables `cross_val_score_parallel` (adds a `rayon`
  dependency).
- `serde` — derives `Serialize` / `Deserialize` on `Split`,
  `CrossValScores`, and `BootstrapCi` for JSON persistence.

## Example

```rust
use ndarray::array;
use solow_cv::{KFold, Splitter};

let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
let kf = KFold::new(5)?.shuffle(false);
for split in kf.split(y.len())? {
    let (train, test) = (split.train, split.test);
    assert_eq!(train.len() + test.len(), y.len());
}
# Ok::<_, solow_core::Error>(())
```

## License

Licensed under either of [Apache License 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
